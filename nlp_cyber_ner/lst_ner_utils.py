"""
LST-NER Utility Functions

Contains utility functions for graph construction and label distribution estimation.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np


def estimate_label_distributions(source_model, dataset, label_ids, num_labels, device, temp=4.0):
    """
    Estimate label probability distributions using a source model
    Returns distribution matrix of shape [num_target_labels, num_source_labels]
    """
    print("Estimating label distributions from source model...")
    source_model.eval()
    
    # Initialize probability distributions
    prob_distributions = torch.zeros(len(label_ids), num_labels)
    sample_counts = torch.zeros(len(label_ids))
    
    # Create a label_id to index mapping
    label_id_to_idx = {label_id: idx for idx, label_id in enumerate(label_ids)}
    
    with torch.no_grad():
        # Process each example in dataset
        for i, sample in enumerate(dataset):
            # Get input tensors
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            labels = sample['labels']
            
            # Get model outputs
            outputs = source_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(0).cpu()  # [seq_len, num_labels]
            
            # Apply temperature scaling
            scaled_logits = logits / temp
            probs = F.softmax(scaled_logits, dim=-1)  # [seq_len, num_labels]
            
            # For each token with a valid label, accumulate distribution
            for j, label in enumerate(labels):
                if label.item() != -100 and label.item() in label_id_to_idx:
                    idx = label_id_to_idx[label.item()]
                    prob_distributions[idx] += probs[j]
                    sample_counts[idx] += 1
    
    # Average the probability distributions
    for i in range(len(label_ids)):
        if sample_counts[i] > 0:
            prob_distributions[i] = prob_distributions[i] / sample_counts[i]
        else:
            # For labels with no samples, use uniform distribution
            prob_distributions[i] = torch.ones(num_labels) / num_labels
    
    print(f"Label distribution estimation complete for {len(label_ids)} labels")
    return prob_distributions


def construct_label_graph(prob_distributions, edge_threshold=0.5):
    """
    Construct a label graph using probability distributions
    Returns adjacency matrix of shape [num_nodes, num_nodes]
    """
    print(f"Constructing label graph with threshold {edge_threshold}...")
    num_nodes = prob_distributions.shape[0]
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    
    # Normalize node representations
    norm_factor = 0.0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance = F.pairwise_distance(
                    prob_distributions[i].unsqueeze(0), 
                    prob_distributions[j].unsqueeze(0), 
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
                    prob_distributions[i].unsqueeze(0), 
                    prob_distributions[j].unsqueeze(0), 
                    p=2
                ).item() / norm_factor
                
                # Add edge if distance is below threshold
                if distance < edge_threshold:
                    similarity = 1.0 - distance / edge_threshold
                    adj_matrix[i, j] = similarity
    
    print(f"Label graph constructed with shape: {adj_matrix.shape}")
    return adj_matrix


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    print(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)