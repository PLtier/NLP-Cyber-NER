"""
LST-NER Dataset Classes

Contains dataset classes for loading and processing NER data.
"""

import torch
from torch.utils.data import Dataset


def convert_bioes_to_bio(tag):
    """Convert BIOES format tags to BIO format"""
    if tag == 'O' or tag == '':
        return tag
    
    # Split into prefix and entity type
    parts = tag.split('-', 1)
    if len(parts) != 2:
        return tag  # Return as is if not in expected format
    
    prefix, entity_type = parts
    
    # Convert BIOES to BIO
    if prefix == 'S':  # Single token entity
        return f"B-{entity_type}"
    elif prefix == 'E':  # End of entity
        return f"I-{entity_type}"
    else:  # B- and I- stay the same
        return tag


class NERDataset(Dataset):
    """Dataset class for NER data in CoNLL format."""
    
    def __init__(self, file_path, tokenizer, labels_to_id, max_length=128):
        self.texts = []
        self.tags = []
        
        # Read CoNLL-like format file
        with open(file_path, 'r', encoding='utf-8') as f:
            current_words = []
            current_tags = []
            
            for line in f:
                line = line.strip()
                if line == '':
                    # End of sentence
                    if current_words:
                        self.texts.append(current_words)
                        self.tags.append(current_tags)
                        current_words = []
                        current_tags = []
                else:
                    # Token and tag
                    parts = line.split(' ')
                    if len(parts) >= 2:
                        token, tag = parts[0], parts[1]
                        # Convert BIOES to BIO format
                        bio_tag = convert_bioes_to_bio(tag)
                        current_words.append(token)
                        current_tags.append(bio_tag)
            
            # Add the last sentence if not empty
            if current_words:
                self.texts.append(current_words)
                self.tags.append(current_tags)
        
        self.tokenizer = tokenizer
        self.labels_to_id = labels_to_id
        self.max_length = max_length
        print(f"Loaded {len(self.texts)} sentences from {file_path}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        words = self.texts[idx]
        tags = self.tags[idx]
        
        # Tokenize the words and align the tags
        encoded_inputs = self.tokenizer(words,
                                      is_split_into_words=True,
                                      max_length=self.max_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors='pt')
        
        # Extract features
        input_ids = encoded_inputs['input_ids'].squeeze(0)
        attention_mask = encoded_inputs['attention_mask'].squeeze(0)
        
        # Convert tags to label ids
        labels = torch.ones(self.max_length, dtype=torch.long) * -100 # -100 is ignored in loss
        
        # Align tags with tokenized words
        word_ids = encoded_inputs.word_ids()
        previous_word_idx = None
        label_idx = 0
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                # Skip special tokens and continuation tokens
                continue
            
            if label_idx < len(tags):
                tag = tags[label_idx]
                labels[i] = self.labels_to_id.get(tag, 0)  # Default to 'O' (0) if unknown
                label_idx += 1
            
            previous_word_idx = word_idx
        
        # Convert lists to tensors or strings for proper batching
        # Store original text and tags as strings to avoid variable length lists
        text_str = ' '.join(words)
        tags_str = ' '.join(tags)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'text': text_str,  # String instead of list
            'tags': tags_str   # String instead of list
        }


def load_raw_data(file_path):
    """Load raw data from CoNLL format file."""
    raw_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        current_words = []
        current_tags = []
        for line in f:
            line = line.strip()
            if line == '':
                if current_words:
                    raw_data.append((current_words, current_tags))
                    current_words = []
                    current_tags = []
            else:
                parts = line.split(' ')
                if len(parts) >= 2:
                    token, tag = parts[0], parts[1]
                    current_words.append(token)
                    current_tags.append(tag)
        if current_words:
            raw_data.append((current_words, current_tags))
    
    return raw_data


def save_deduplicated_data(raw_data, output_path):
    """Save deduplicated data to file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for words, tags in raw_data:
            for word, tag in zip(words, tags):
                bio_tag = convert_bioes_to_bio(tag)
                f.write(f"{word} {bio_tag}\n")
            f.write("\n")


def extract_labels_from_file(file_path):
    """Extract all unique labels from a CoNLL format file."""
    labels = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and len(line.split(' ')) >= 2:
                _, label = line.split(' ')[:2]
                # Convert to BIO format for label collection
                bio_label = convert_bioes_to_bio(label)
                labels.add(bio_label)
    
    return labels