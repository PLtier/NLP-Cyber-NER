"""
LST-NER Evaluation Functions

Contains evaluation functions for the LST-NER model.
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from .span_f1 import span_f1


def evaluate_model(model, eval_dataloader, device, id2label=None):
    """
    Evaluate the model on a validation set using span-based F1 metrics
    
    Args:
        model: The model to evaluate
        eval_dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        id2label: Dictionary mapping from label ID (as string) to label name
    """
    # Create a fallback id2label mapping if none is provided
    if id2label is None:
        print("Warning: No id2label dictionary provided, using default mapping")
        id2label = {
            "0": "O", 
            "1": "B-Entity", "2": "I-Entity",
            "3": "B-Other", "4": "I-Other"
        }
    print("Evaluating model...")
    model.eval()
    
    # For token-level metrics
    all_token_preds = []
    all_token_labels = []
    
    # For span-based metrics
    all_pred_tags = []
    all_gold_tags = []
    current_pred_tags = []
    current_gold_tags = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = model(**batch)
            logits = outputs['logits']
            
            # Get predictions
            preds = torch.argmax(logits, dim=2)
            
            # Collect predictions and labels, excluding padded tokens (-100)
            labels = batch['labels']
            
            for i in range(labels.shape[0]):  # Iterate over batch
                current_pred_tags = []
                current_gold_tags = []
                
                for j in range(labels.shape[1]):  # Iterate over sequence
                    if labels[i, j] != -100:  # Exclude padding tokens
                        all_token_preds.append(preds[i, j].item())
                        all_token_labels.append(labels[i, j].item())
                        
                        # Convert numeric labels to BIO tags for span evaluation
                        try:
                            if id2label:
                                pred_tag = id2label[str(preds[i, j].item())]
                                gold_tag = id2label[str(labels[i, j].item())]
                            else:
                                # Default conversion if id2label not provided
                                if preds[i, j].item() == 0:
                                    pred_tag = 'O'
                                else:
                                    tag_type = 'B' if preds[i, j].item() % 2 == 1 else 'I'
                                    tag_class = f"-{(preds[i, j].item() + 1) // 2}"
                                    pred_tag = tag_type + tag_class
                                    
                                if labels[i, j].item() == 0:
                                    gold_tag = 'O'
                                else:
                                    tag_type = 'B' if labels[i, j].item() % 2 == 1 else 'I'
                                    tag_class = f"-{(labels[i, j].item() + 1) // 2}"
                                    gold_tag = tag_type + tag_class
                        except Exception as e:
                            print(f"Error converting labels to tags: {e}")
                            # Fallback to simple O tag if there's an error
                            pred_tag = 'O'
                            gold_tag = 'O'
                        
                        current_pred_tags.append(pred_tag)
                        current_gold_tags.append(gold_tag)
                    else:
                        # End of sentence
                        if current_pred_tags and current_gold_tags:
                            break
                            
                if current_pred_tags and current_gold_tags:
                    all_pred_tags.append(current_pred_tags)
                    all_gold_tags.append(current_gold_tags)
    
    # Calculate token-level metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_token_labels, all_token_preds, average='weighted', zero_division=1
    )
    
    # Calculate accuracy
    accuracy = (np.array(all_token_preds) == np.array(all_token_labels)).mean()
    
    # Calculate span-based metrics
    span_metrics = span_f1(all_gold_tags, all_pred_tags)
    
    # Print both metrics for comparison
    print(f"Token-level metrics: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Accuracy={accuracy:.4f}")
    print(f"Span-based metrics: F1={span_metrics['slot-f1']:.4f}, Precision={span_metrics['precision']:.4f}, Recall={span_metrics['recall']:.4f}")
    
    return {
        'token_precision': precision,
        'token_recall': recall,
        'token_f1': f1,
        'token_accuracy': accuracy,
        'span_precision': span_metrics['precision'],
        'span_recall': span_metrics['recall'],
        'span_f1': span_metrics['slot-f1'],
        'f1': span_metrics['slot-f1']  # Use span F1 as the primary metric
    }