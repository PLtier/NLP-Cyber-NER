"""
LST-NER: Label Structure Transfer for Named Entity Recognition

Main training script for LST-NER model.
"""

import os
import sys
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Import our modules
from .lst_ner_model import LSTNER
from ..lst_ner_dataset import NERDataset, load_raw_data, save_deduplicated_data, extract_labels_from_file
from ..lst_ner_utils import estimate_label_distributions, construct_label_graph, set_seed
from ..lst_ner_evaluation import evaluate_model
from ..dataset import remove_leakage


def save_predictions(model, dataloader, device, id2label, output_path, dataset_name):
    """
    Save model predictions to file in the format expected by lookup tools
    
    Args:
        model: Trained model
        dataloader: DataLoader for the dataset
        device: Device to run inference on
        id2label: Label mapping dictionary
        output_path: Directory to save predictions
        dataset_name: Name of dataset (e.g., 'dev', 'test')
    """
    import os
    
    model.eval()
    os.makedirs(output_path, exist_ok=True)
    
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Generating predictions for {dataset_name}"):
            # Move batch to device
            batch_tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = model(**batch_tensors)
            logits = outputs['logits']
            
            # Get predictions
            preds = torch.argmax(logits, dim=2)
            labels = batch_tensors['labels']
            
            # Process each example in batch
            for i in range(len(batch['text'])):
                # Get original text and labels
                text_tokens = batch['text'][i].split()
                true_labels = batch['tags'][i].split()
                
                # Get predictions for this example (only for non-padded tokens)
                example_preds = preds[i]
                example_labels = labels[i]
                pred_labels = []
                filtered_true_labels = []
                filtered_tokens = []
                
                # Only include tokens that aren't padded (-100)
                token_idx = 0
                for j in range(len(example_labels)):
                    if example_labels[j] != -100 and token_idx < len(text_tokens):
                        pred_label_id = example_preds[j].item()
                        pred_label = id2label.get(str(pred_label_id), 'O')
                        pred_labels.append(pred_label)
                        filtered_true_labels.append(true_labels[token_idx] if token_idx < len(true_labels) else 'O')
                        filtered_tokens.append(text_tokens[token_idx])
                        token_idx += 1
                
                # Store prediction in CoNLL format
                for token, true_label, pred_label in zip(filtered_tokens, filtered_true_labels, pred_labels):
                    predictions.append(f"{token} {true_label} {pred_label}")
                predictions.append("")  # Empty line between sentences
    
    # Save predictions to file
    pred_file = os.path.join(output_path, f"{dataset_name}_predictions.txt")
    with open(pred_file, 'w', encoding='utf-8') as f:
        for line in predictions:
            f.write(line + '\n')
    
    print(f"Predictions saved to: {pred_file}")


def main():
    try:
        print("Starting LST-NER script...")
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Configuration
        config = {
            "base_model_name": "bert-base-cased",
            "max_length": 128,
            "batch_size": 16,
            "learning_rate": 5e-5,
            "epochs": 3,
            "output_dir": "lst_ner_output",
            "temp": 4.0,  # Temperature for distribution smoothing
            "edge_threshold": 1.5,  # Threshold for adding edges in the label graph
            "gwd_lambda": 0.01,  # Weight for GWD loss
            "target_train_data": "data/processed/YOUR_FOLDER/YOUR_TRAIN_DATASET.unified",
            "target_dev_data": "data/processed/YOUR_FOLDER/YOUR_dev_DATASET.unified",
            "target_test_data": "data/processed/YOUR_FOLDER/YOUR_test_DATASET.unified",
            "predictions_output_dir": "artifacts/predictions/lst_ner/"
        }
        
        print("Configuration set successfully")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA device count:", torch.cuda.device_count())
            print("CUDA current device:", torch.cuda.current_device())
            print("CUDA device name:", torch.cuda.get_device_name(0))
        
        # Load real datasets and pre-trained NER model
        print("Loading datasets and pre-trained NER model...")
        
        # Load pre-trained NER model from Hugging Face (CoNLL-2003 English)
        print("Loading pre-trained NER model...")
        source_model_name = "dslim/bert-base-NER"  # Trained on CoNLL-2003 (English)
        source_model = AutoModelForTokenClassification.from_pretrained(source_model_name)
        source_model.to(device)
        
        # Get source labels from the config
        source_config = source_model.config
        num_source_labels = source_config.num_labels
        source_labels = source_config.id2label if hasattr(source_config, 'id2label') else None
        
        print(f"Pre-trained NER model loaded: {source_model_name}")
        print(f"Number of source labels: {num_source_labels}")
        if source_labels:
            print(f"Source labels: {list(source_labels.values())}")
        
        # Load real your dataset
        print("Loading your datasets...")
        tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"])
        
        # Extract all labels from the your dataset and convert to BIO
        print("Extracting target labels from your dataset and converting to BIO format...")
        all_labels = set()
        
        # Read training file to extract labels
        all_labels.update(extract_labels_from_file(config["target_train_data"]))
        # Read dev file to extract additional labels (if any)
        all_labels.update(extract_labels_from_file(config["target_dev_data"]))
        
        # Sort labels to ensure consistent ordering, with 'O' as the first label
        target_labels = ['O'] if 'O' in all_labels else []
        sorted_labels = sorted(all_labels - {'O'} if 'O' in all_labels else all_labels)
        target_labels.extend(sorted_labels)
        
        print(f"Extracted {len(target_labels)} unique labels from the dataset")
        print(f"Labels: {target_labels}")
        
        # Create label mappings
        target_id2label = {str(i): label for i, label in enumerate(target_labels)}
        labels_to_id = {label: i for i, label in enumerate(target_labels)}  # Use this name for NERDataset
        
        # Store in config for later use
        config["id2label"] = target_id2label
        
        # Load raw data for deduplication
        print("Loading raw data for deduplication...")
        raw_train_data = load_raw_data(config["target_train_data"])
        raw_dev_data = load_raw_data(config["target_dev_data"])
                
        # Apply deduplication to remove overlapping examples
        print(f"Before deduplication - Train: {len(raw_train_data)}, Dev: {len(raw_dev_data)} examples")
        raw_train_data, removed_train_examples = remove_leakage(raw_train_data, raw_dev_data)
        print(f"After deduplication - Train: {len(raw_train_data)}, Dev: {len(raw_dev_data)} examples")
        print(f"Removed {len(removed_train_examples)} overlapping examples from train dataset")
        
        # Convert the deduplicated data back to temporary files
        deduplicated_train_path = config["target_train_data"] + ".deduplicated"
        save_deduplicated_data(raw_train_data, deduplicated_train_path)
        
        # Create datasets using deduplicated data
        train_dataset = NERDataset(deduplicated_train_path, tokenizer, labels_to_id, config["max_length"])
        dev_dataset = NERDataset(config["target_dev_data"], tokenizer, labels_to_id, config["max_length"])
        
        # Create dataloaders
        print("Creating dataloaders...")
        train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=config["batch_size"], shuffle=False)
        print("Dataloaders created")
        
        # Estimate label distributions
        print("Estimating label distributions...")
        num_target_labels = len(target_labels)
        label_ids = list(range(num_target_labels))
        prob_distributions = estimate_label_distributions(
            source_model,
            train_dataset,
            label_ids,
            num_source_labels,
            device,
            temp=config["temp"]
        )
        print(f"Created label distributions with shape: {prob_distributions.shape}")
        
        # Construct source graph
        print("Constructing source graph...")
        source_graph = construct_label_graph(
            prob_distributions,
            edge_threshold=config["edge_threshold"]
        )
        print(f"Source graph constructed with shape: {source_graph.shape}")
        
        # Create LST-NER model with GWD
        print("Creating LST-NER model with GWD...")
        lst_ner_model = LSTNER(
            base_model_name=config["base_model_name"],
            target_labels=target_labels,
            temp=config["temp"],
            edge_threshold=config["edge_threshold"],
            gwd_lambda=config["gwd_lambda"]
        )
        
        # Set source graph
        print("Setting source graph...")
        lst_ner_model.set_source_graph(source_graph)
        
        # Perform a test forward pass
        print("Testing forward pass...")
        lst_ner_model.to(device)
        batch = next(iter(train_dataloader))
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            outputs = lst_ner_model(**batch)
        print(f"Forward pass successful, output shapes: {[k + ': ' + str(v.shape) if isinstance(v, torch.Tensor) else k + ': N/A' for k, v in outputs.items()]}")
        
        # Full training setup
        print("\nStarting full training...")
        
        # Prepare model for training
        optimizer = AdamW(
            lst_ner_model.parameters(),
            lr=config["learning_rate"],
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Calculate total training steps for scheduler
        num_training_steps = len(train_dataloader) * config["epochs"]
        num_warmup_steps = int(num_training_steps * 0.1)  # 10% of total steps for warmup
        
        # Create learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Training tracking variables
        best_f1 = 0.0
        steps_no_improve = 0
        early_stop_patience = 3  # Stop if no improvement for 3 epochs
        epochs_completed = 0
        
        # Create directory for saving models
        os.makedirs(config["output_dir"], exist_ok=True)
        
        print(f"\nTraining for {config['epochs']} epochs with {len(train_dataloader)} batches per epoch")
        print(f"Total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
        print(f"Batch size: {config['batch_size']}, Learning rate: {config['learning_rate']}")
        print(f"GWD lambda: {config['gwd_lambda']}, Edge threshold: {config['edge_threshold']}")
        
        # Training loop
        for epoch in range(config["epochs"]):
            print(f"\n{'='*80}\nStarting Epoch {epoch+1}/{config['epochs']}\n{'='*80}")
            lst_ner_model.train()
            
            # Training metrics
            epoch_loss = 0
            epoch_gwd_loss = 0
            num_batches = 0
            
            # Process each batch
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = lst_ner_model(**batch)
                
                # Get loss components
                loss = outputs['loss']
                gwd_loss = outputs['gwd_loss']
                
                # Backward pass
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(lst_ner_model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                current_loss = loss.item()
                epoch_loss += current_loss
                num_batches += 1
                
                if gwd_loss is not None:
                    current_gwd = gwd_loss.item()
                    epoch_gwd_loss += current_gwd
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{current_loss:.4f}", 
                        'gwd_loss': f"{current_gwd:.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.6f}"
                    })
                else:
                    # Update progress bar without GWD loss
                    progress_bar.set_postfix({
                        'loss': f"{current_loss:.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.6f}"
                    })
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / num_batches
            avg_gwd_loss = epoch_gwd_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}, Average GWD loss: {avg_gwd_loss:.4f}")
            
            # Evaluate on dev set
            print("\nEvaluating on dev set...")
            # Use the id2label mapping from config
            eval_results = evaluate_model(lst_ner_model, dev_dataloader, device, config["id2label"])
            print(f"Dev set results: F1={eval_results['span_f1']:.4f}, Precision={eval_results['span_precision']:.4f}, Recall={eval_results['span_recall']:.4f}")
            
            # Save model if it's the best so far
            current_f1 = eval_results['span_f1']  # Use span-based F1
            if current_f1 > best_f1:
                print(f"New best F1: {current_f1:.4f} (previous: {best_f1:.4f}). Saving model...")
                best_f1 = current_f1
                steps_no_improve = 0
                
                # Save the best model
                best_model_path = os.path.join(config["output_dir"], "best_model")
                lst_ner_model.save_pretrained(best_model_path)
                
                # Save training info
                with open(os.path.join(config["output_dir"], "training_info.json"), "w") as f:
                    json.dump({
                        "best_f1": best_f1,
                        "best_epoch": epoch + 1,
                        "span_precision": eval_results['span_precision'],
                        "span_recall": eval_results['span_recall'],
                        "token_precision": eval_results['token_precision'],
                        "token_recall": eval_results['token_recall'],
                        "token_accuracy": eval_results['token_accuracy'],
                        "config": config
                    }, f, indent=2)
            else:
                steps_no_improve += 1
                print(f"No improvement for {steps_no_improve} epochs. Best F1: {best_f1:.4f}")
            
            # Early stopping check
            if steps_no_improve >= early_stop_patience:
                print(f"\nEarly stopping after {epoch+1} epochs without improvement")
                break
            
            epochs_completed = epoch + 1
        
        print(f"\nTraining completed after {epochs_completed} epochs.")
        print(f"Best F1 score: {best_f1:.4f}")
        
        # Load the best model for final evaluation
        print("\nLoading best model for final evaluation...")
        best_model_path = os.path.join(config["output_dir"], "best_model")
        if os.path.exists(os.path.join(best_model_path, "pytorch_model.bin")):
            # Load the best model state dict
            best_state_dict = torch.load(os.path.join(best_model_path, "pytorch_model.bin"))
            lst_ner_model.load_state_dict(best_state_dict)
            
            # Final evaluation
            print("\nFinal evaluation on dev set:")
            final_eval_results = evaluate_model(lst_ner_model, dev_dataloader, device, target_id2label)
            print(f"Final evaluation results: Span F1={final_eval_results['span_f1']:.4f}, Precision={final_eval_results['span_precision']:.4f}, Recall={final_eval_results['span_recall']:.4f}")
        else:
            print("Best model not found. Using last model for final evaluation.")
            final_eval_results = eval_results
            
        print("\nTraining and evaluation complete!")
        
        # Save predictions for dev set
        print("\nSaving predictions for dev set...")
        save_predictions(
            lst_ner_model,
            dev_dataloader, 
            device, 
            config["id2label"], 
            config["predictions_output_dir"], 
            "dev"
        )
        
        # If test data exists, save predictions for test set
        if "target_test_data" in config and os.path.exists(config["target_test_data"]):
            print("Loading test dataset...")
            test_dataset = NERDataset(config["target_test_data"], tokenizer, labels_to_id, config["max_length"])
            test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
            
            print("Saving predictions for test set...")
            save_predictions(
                lst_ner_model,
                test_dataloader, 
                device, 
                config["id2label"], 
                config["predictions_output_dir"], 
                "test"
            )
        
        print("\nScript execution complete")
        
    except Exception as e:
        print(f"ERROR: An exception occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nScript execution failed with error")


if __name__ == "__main__":
    main()