"""
LST-NER: Label Structure Transfer for Named Entity Recognition Model

Contains the core LSTNER model class and related graph components.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    import ot
except ImportError:
    print("Warning: POT (Python Optimal Transport) package not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "POT"])
    import ot

from transformers import AutoModel


class GraphConvolution(nn.Module):
    """Graph Convolutional Network layer for processing label graphs."""
    
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        print(f"Initializing GraphConvolution layer with in_features={in_features}, out_features={out_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input, adj):
        """
        input: node features [batch_size, num_nodes, in_features]
        adj: adjacency matrix [num_nodes, num_nodes]
        """
        # Normalize adjacency matrix
        adj = self._normalize_adj(adj)
        
        # Message passing
        support = torch.matmul(input, self.weight)  # [batch_size, num_nodes, out_features]
        output = torch.matmul(adj, support)  # [batch_size, num_nodes, out_features]
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix"""
        # Replace inf with 0
        adj = torch.where(adj == float('inf'), torch.zeros_like(adj), adj)
        
        # Add self-loops
        identity = torch.eye(adj.size(0), device=adj.device)
        adj = adj + identity
        
        # Calculate degree matrix
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        # Normalize
        return torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)


class LSTNER(nn.Module):
    """LST-NER model with target graph and Gromov-Wasserstein Distance."""
    
    def __init__(self, base_model_name, target_labels, temp=4.0, edge_threshold=0.5, gwd_lambda=0.01):
        super(LSTNER, self).__init__()
        print(f"Initializing LSTNER with {len(target_labels)} labels and edge_threshold={edge_threshold}")
        
        # BERT backbone
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.hidden_dim = self.bert.config.hidden_size
        self.dp_dim = self.hidden_dim  # Projection dimension
        
        # Target labels
        self.target_labels = target_labels
        self.num_target_labels = len(target_labels)
        
        # Label semantic fusion layers
        self.Wp = nn.Linear(self.hidden_dim, self.dp_dim)
        self.bp = nn.Parameter(torch.zeros(self.dp_dim))
        
        # Label representation (randomly initialized)
        self.label_embeddings = nn.Parameter(torch.randn(self.num_target_labels, self.dp_dim))
        
        # GCN for label graph structure
        self.gcn = GraphConvolution(self.dp_dim, self.dp_dim)
        
        # Projection layers for label fusion
        self.W_prime = nn.Linear(self.dp_dim, self.hidden_dim)
        self.b_prime = nn.Parameter(torch.zeros(self.hidden_dim))
        
        # Task-specific classifiers
        self.classifier = nn.Linear(self.hidden_dim, self.num_target_labels)
        self.aux_classifier = nn.Linear(self.hidden_dim, self.num_target_labels) 
        
        # Source and target graphs
        self.source_graph = None
        self.target_graph = None
        self.target_distributions = None
        
        # Tracking target distribution accumulation
        self.target_dist_accum = torch.zeros(self.num_target_labels, self.num_target_labels)
        self.target_dist_count = torch.zeros(self.num_target_labels)
        
        # Hyperparameters
        self.temp = temp
        self.edge_threshold = edge_threshold
        self.gwd_lambda = gwd_lambda  # Weight for GWD loss
        
    def set_source_graph(self, source_graph):
        """Set the source graph for label structure transfer"""
        self.source_graph = source_graph
    
    def update_target_distributions(self, logits, labels, update_graph=False):
        """
        Update target label distributions based on model predictions
        
        Args:
            logits: Model predictions [batch_size, seq_len, num_target_labels]
            labels: Ground truth labels [batch_size, seq_len]
            update_graph: Whether to update the target graph after updating distributions
        """
        batch_size, seq_len, _ = logits.shape
        device = logits.device
        
        # Apply temperature scaling to logits
        scaled_logits = logits / self.temp
        probs = F.softmax(scaled_logits, dim=-1)  # [batch_size, seq_len, num_target_labels]
        
        # Accumulate probability distributions for each label
        for b in range(batch_size):
            for s in range(seq_len):
                label = labels[b, s].item()
                if label != -100 and 0 <= label < self.num_target_labels:
                    self.target_dist_accum[label] += probs[b, s].detach().cpu()
                    self.target_dist_count[label] += 1
        
        # If we have enough samples and update_graph is True, update target graph
        if update_graph and torch.any(self.target_dist_count > 0):
            self.compute_target_graph()
    
    def compute_target_graph(self):
        """Compute target graph from accumulated distributions"""
        # Average the accumulated distributions
        target_distributions = torch.zeros_like(self.target_dist_accum)
        for i in range(self.num_target_labels):
            if self.target_dist_count[i] > 0:
                target_distributions[i] = self.target_dist_accum[i] / self.target_dist_count[i]
            else:
                # For labels with no samples, use uniform distribution
                target_distributions[i] = torch.ones(self.num_target_labels) / self.num_target_labels
        
        # Store the distributions
        self.target_distributions = target_distributions
        
        # Construct the target graph
        print("Constructing target graph...")
        self.target_graph = self.construct_graph_from_distributions(target_distributions)
        print(f"Target graph constructed with shape: {self.target_graph.shape}")
    
    def construct_graph_from_distributions(self, distributions):
        """Construct a graph from label distributions"""
        num_nodes = distributions.shape[0]
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        
        # Normalize node representations
        norm_factor = 0.0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    distance = F.pairwise_distance(
                        distributions[i].unsqueeze(0), 
                        distributions[j].unsqueeze(0), 
                        p=2
                    ).item()
                    norm_factor += distance
        
        if norm_factor > 0:
            norm_factor = norm_factor / (num_nodes * (num_nodes - 1))
        else:
            norm_factor = 1.0
        
        # Calculate similarities and construct graph
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    # Self-loop
                    adj_matrix[i, j] = 1.0
                else:
                    # Calculate normalized distance
                    distance = F.pairwise_distance(
                        distributions[i].unsqueeze(0), 
                        distributions[j].unsqueeze(0), 
                        p=2
                    ).item() / norm_factor
                    
                    # Add edge if distance is below threshold
                    if distance < self.edge_threshold:
                        similarity = 1.0 - distance / self.edge_threshold
                        adj_matrix[i, j] = similarity
        
        return adj_matrix
    
    def compute_gwd(self, source_graph, target_graph):
        """
        Compute the Gromov-Wasserstein Distance between source and target graphs
        
        Returns:
            gwd_loss: The GWD loss value
        """
        # Convert PyTorch tensors to NumPy for POT library
        source_adj = source_graph.cpu().numpy()
        target_adj = target_graph.cpu().numpy()
        
        # Create uniform distributions for graph nodes
        p = np.ones(source_adj.shape[0]) / source_adj.shape[0]
        q = np.ones(target_adj.shape[0]) / target_adj.shape[0]
        
        # Compute the Gromov-Wasserstein distance
        # Note: We use the squared Euclidean loss for structure matching
        gwd = ot.gromov.gromov_wasserstein2(
            source_adj, target_adj, p, q, 'square_loss', verbose=False
        )
        
        return torch.tensor(gwd, device=source_graph.device)
        
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass with label-guided attention and GCN"""
        # Get BERT contextual embeddings
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        token_embeds = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
        batch_size, seq_len, _ = token_embeds.shape
        
        # Check if source graph exists
        if self.source_graph is None:
            print("Warning: Source graph not set, using identity matrix")
            self.source_graph = torch.eye(self.num_target_labels)
        
        # Ensure source graph is on the correct device
        source_graph = self.source_graph.to(input_ids.device)
        
        # Label-guided attention for extracting label-specific components
        q = self.Wp(token_embeds) + self.bp  # [batch_size, seq_len, dp_dim]
        
        # Calculate label-specific components for each sentence
        label_specific_comps = torch.zeros(batch_size, self.num_target_labels, self.dp_dim, device=input_ids.device)
        
        for i in range(batch_size):
            # Calculate attention weights between tokens and label embeddings
            alpha = F.softmax(torch.matmul(q[i], self.label_embeddings.transpose(0, 1)), dim=0)  # [seq_len, num_target_labels]
            
            # Extract label-specific components
            for l in range(self.num_target_labels):
                label_specific_comps[i, l] = torch.matmul(alpha[:, l], q[i])  # [dp_dim]
        
        # Apply GCN to enhance label-specific components with graph structure
        enhanced_label_comps = self.gcn(label_specific_comps, source_graph)
        
        # Token-guided attention to fuse label-specific components into token embeddings
        enhanced_token_embeds = token_embeds.clone()
        
        for i in range(batch_size):
            for j in range(seq_len):
                # Calculate attention weights
                beta = F.softmax(torch.matmul(q[i, j], enhanced_label_comps[i].transpose(0, 1)), dim=0)  # [num_target_labels]
                # Fuse label-specific components
                weighted_sum = torch.matmul(beta, enhanced_label_comps[i])  # [dp_dim]
                enhanced_token_embeds[i, j] += self.W_prime(weighted_sum) + self.b_prime
        
        # Classification for NER
        logits = self.classifier(enhanced_token_embeds)  # [batch_size, seq_len, num_target_labels]
        
        # Auxiliary task for entity type detection (sentence level)
        # Use average pooling for sequence representation
        avg_token_embeds = torch.mean(enhanced_token_embeds, dim=1)  # [batch_size, hidden_dim]
        aux_logits = self.aux_classifier(avg_token_embeds)  # [batch_size, num_target_labels]
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # NER token classification loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            token_loss = loss_fct(logits.view(-1, self.num_target_labels), labels.view(-1))
            
            # Auxiliary loss (entity type detection)
            # Since we don't have sentence-level entity type labels in this implementation,
            # we'll create approximate labels: a label is present if any token has that label
            aux_labels = torch.zeros(batch_size, self.num_target_labels, device=labels.device)
            for i in range(batch_size):
                for j in range(seq_len):
                    if labels[i, j] >= 0 and labels[i, j] < self.num_target_labels:
                        aux_labels[i, labels[i, j]] = 1.0
            
            aux_loss = F.binary_cross_entropy_with_logits(aux_logits, aux_labels)
            
            # Update target distributions based on current predictions
            self.update_target_distributions(logits, labels, update_graph=True)
            
            # GWD loss (if both graphs are available)
            gwd_loss = torch.tensor(0.0, device=labels.device)
            if self.source_graph is not None and self.target_graph is not None:
                # Move target graph to current device
                target_graph = self.target_graph.to(labels.device)
                gwd_loss = self.compute_gwd(source_graph, target_graph)
                print(f"GWD Loss: {gwd_loss.item():.4f}")
            
            # Combine losses
            loss = token_loss + 0.1 * aux_loss  # Base loss
            
            # Add GWD loss if available
            if gwd_loss > 0:
                loss = loss + self.gwd_lambda * gwd_loss
        
        return {
            'loss': loss,
            'logits': logits,
            'aux_logits': aux_logits,
            'enhanced_embeds': enhanced_token_embeds,
            'gwd_loss': gwd_loss if 'gwd_loss' in locals() else None
        }
    
    def save_pretrained(self, output_path):
        """Save model to disk"""
        os.makedirs(output_path, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
        
        # Save config
        config = {
            "hidden_dim": self.hidden_dim,
            "dp_dim": self.dp_dim,
            "edge_threshold": self.edge_threshold,
            "temp": self.temp,
            "gwd_lambda": self.gwd_lambda,
            "num_target_labels": self.num_target_labels,
            "target_labels": self.target_labels
        }
        
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f)