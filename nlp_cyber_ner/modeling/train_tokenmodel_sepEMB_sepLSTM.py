# This document just exists to ensure that training happens properly. The performance by these experiments should be compared
# to the token reference model, and should be practically identical


import gc
import json
import copy
import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from itertools import cycle
import random
from pathlib import Path
from IPython.display import clear_output

from nlp_cyber_ner.config import DATA_DIR, PROCESSED_DATA_DIR, TOKENPROCESSED_DATA_DIR, load_dotenv

from nlp_cyber_ner.dataset import (
    read_iob2_file,
    preds_to_tags,
    transform_dataset,
    remove_leakage
)
from nlp_cyber_ner.span_f1 import span_f1

#Hyperparameters
BATCH_SIZE = 32
DIM_EMBEDDING = 100
LSTM_HIDDEN = 100
DROPOUT1 = 0.5
DROPOUT2 = 0.5
LEARNING_RATE = 0.001
EPOCHS = 15
CLIPPING = 5.0 # - didn't seem like it was needed from testing, but used in the original (train.py) model so reusing here


#Determines whether to run train & eval with out without deduplication
DEDUPLICATION = True


#Setting up datasets:
    #CYNER: (not changes for tokenmodel so loaded just from processed directory)
cyner_path = PROCESSED_DATA_DIR / "cyner"
cyner_train_path = cyner_path / "train.unified"
cyner_dev_path = cyner_path / "valid.unified"
cyner_test_path = cyner_path / "test.unified"
cyner_train_data = read_iob2_file(cyner_train_path)
cyner_dev_data = read_iob2_file(cyner_dev_path)
cyner_test_data = read_iob2_file(cyner_test_path)


    #ATTACKNER:
attackner_path = TOKENPROCESSED_DATA_DIR / "attacker"
attackner_train_path  = attackner_path / "train.tokenready"
attackner_dev_path= attackner_path / "dev.tokenready"
attackner_test_path= attackner_path / "test.tokenready"
attackner_train_data = read_iob2_file(attackner_train_path, word_index=0, tag_index=1)
attackner_dev_data = read_iob2_file(attackner_dev_path, word_index=0, tag_index=1)
attackner_test_data = read_iob2_file(attackner_test_path, word_index=0, tag_index=1)


    #APTNER:
aptner_path = TOKENPROCESSED_DATA_DIR / "APTNer"
aptner_train_path= aptner_path / "train.tokenready"
aptner_dev_path= aptner_path / "valid.tokenready"
aptner_test_path= aptner_path / "test.tokenready"
aptner_train_data = read_iob2_file(aptner_train_path)
aptner_dev_data = read_iob2_file(aptner_dev_path)
aptner_test_data = read_iob2_file(aptner_test_path)


    #DNRTI:
dnrti_path = TOKENPROCESSED_DATA_DIR / "DNRTI"
dnrti_train_path = dnrti_path / "train.tokenready"
dnrti_dev_path = dnrti_path / "valid.tokenready"
dnrti_test_path = dnrti_path / "test.tokenready"
dnrti_train_data = read_iob2_file(dnrti_train_path, word_index=0, tag_index=1)
dnrti_dev_data = read_iob2_file(dnrti_dev_path, word_index=0, tag_index=1)
dnrti_test_data = read_iob2_file(dnrti_test_path, word_index=0, tag_index=1)


#names of the current datasets
DATASETS = ["dnrti", "aptner", "attackner", "cyner"]

DATASETS_DATAPACK_TRAIN = { #uses the data structures resulting from the red_iob2_file function - i.e. list of tuples of lists.
    "dnrti": dnrti_train_data, 
    "aptner": aptner_train_data, 
    "attackner": attackner_train_data, 
    "cyner": cyner_train_data}

DATASETS_DATAPACK_DEV = {
    "dnrti": dnrti_dev_data, 
    "aptner": aptner_dev_data, 
    "attackner": attackner_dev_data, 
    "cyner": cyner_dev_data
}

DATASETS_DATAPACK_TEST = {
    "dnrti": dnrti_test_data, 
    "aptner": aptner_test_data, 
    "attackner": attackner_test_data, 
    "cyner": cyner_test_data
}


#Helper function for tokenmodel to build dictionaries for vocab and label sizes - used in initalization of tokenmodel dimensions
#Specifically, vocabulary sizes are for the embedding matrices, and label set sizes are for the output heads.
def buildtokenmodeldicts(current_datasets, vocabulary_sizes, label_set_sizes):
    """
    current datasets (list): list of dataset names 
    vocabulary sizes & label_set_sizes (lists): corresponding vocabulary sizes and label set sizes for the named datasets.
    """
    label_size_dict = {}
    vocab_size_dict = {}
    for i, dataset_name in enumerate(current_datasets):
        label_size_dict[dataset_name] = label_set_sizes[i]
        vocab_size_dict[dataset_name] = vocabulary_sizes[i]
    return label_size_dict, vocab_size_dict


def getBatchSamplingProbs(train_loaders: dict):
    total_batches = sum(len(loader) for loader in train_loaders.values())
    datasets = list(train_loaders.keys())

    dataset_sampling_probabilities = []
    print("total batches ", total_batches)
    for dataset_name in datasets:
        print(dataset_name, " Individial batches: ", len(train_loaders[dataset_name]))
        dataset_sampling_probabilities.append(len(train_loaders[dataset_name]) / total_batches)

    print("Probabilities to sample from the datasets: ", dataset_sampling_probabilities)

    return total_batches, dataset_sampling_probabilities

#Tokenmodel:
class TokenModel_sepEMB_tiedLSTM(nn.Module):
    def __init__(self, dataset_vocab_sizes, dataset_label_map):
        super().__init__()
        # instead of a single embedding layer, we create one per dataset - might be better to create a common vocabulary, but this is 
        # definitely way easier given the scripts we have so far.
        self.embeddings = nn.ModuleDict()
        for dataset_name, vocab_size in dataset_vocab_sizes.items():
            self.embeddings[dataset_name] = nn.Embedding(vocab_size, DIM_EMBEDDING)

        self.drop1 = nn.Dropout(p=DROPOUT1)
    
        self.rnns = nn.ModuleDict()
        for dataset_name, _ in dataset_label_map.items():
            self.rnns[dataset_name] = nn.LSTM(DIM_EMBEDDING, LSTM_HIDDEN, batch_first=True, bidirectional=True)

        self.drop2 = nn.Dropout(p=DROPOUT2)
        
        # create one output head per dataset
        self.heads = nn.ModuleDict()
        for dataset_name, ntags in dataset_label_map.items():
            self.heads[dataset_name] = nn.Linear(LSTM_HIDDEN * 2, ntags)

    def forward(self, input_data, dataset_name):

        # Retrieve the dataset-specific embedding layer. Returns error if dataset_name given to model doesn't match those given during initialization
        if dataset_name not in self.embeddings:
            raise ValueError(f"Dataset id '{dataset_name}' not recognized. "
                             f"Available embeddings: {list(self.embeddings.keys())}")
        embed_layer = self.embeddings[dataset_name]
        word_vectors = embed_layer(input_data) 
        
        regular1 = self.drop1(word_vectors)

        if dataset_name not in self.rnns:
            raise ValueError(f"Dataset id '{dataset_name}' not recognized for LSTM cells. "
                             f"LSTM cells available for: {list(self.rnns.keys())}")
        output, hidden = self.rnns[dataset_name](regular1)        

        regular2 = self.drop2(output)
                
        # Retrieve the dataset-specific head. Returns error if dataset_name given to model doesn't match those given during initialization
        if dataset_name not in self.heads:
            raise ValueError(f"Dataset id '{dataset_name}' not recognized in heads. "
                             f"Available heads: {list(self.heads.keys())}")
        predictions = self.heads[dataset_name](regular2)  # [batch_size, max_len, num_labels]
        return predictions



def train_tokenmodel(model, train_loaders, total_batches, sampling_probs, epochs, device, datasets, max_grad_norm) -> TokenModel_sepEMB_tiedLSTM:

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    torch.manual_seed(0)

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch + 1}\n-------------------------------")
        allsets_epoch_loss = 0.0

        # Create infinite iterators for each loader using cycle() - in case a dataset happens to get exhausted during an epoch.
        # This is done inside the epoch loop so that it is initalized during each epoch. This is because when it is exhausted, and
        # cycle is used, the iterable is not randomly initalized again - it uses a cached, copied version, meaning the order of the batches
        # would be the same again.
        loader_iters = {dataset_name: cycle(loader) for dataset_name, loader in train_loaders.items()}

        #https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice, according to this thread, sampling a large number
        #of instances once, is much faster than sampling one at a a time (even if you use python's random.choice)
        #- so we sample all the dataset_names for batch selection in the epoch here, rather than inside the nested loop.
        sampled_dataset_names_for_epoch = np.random.choice(datasets, size=total_batches, p=sampling_probs)

        #just setting up some loss tracking variables:
        dataset_losses = {}
        datasets_loss_history = {}
        for dataset_name in datasets:
            dataset_losses[dataset_name] = [0.0, 0.0, 0, 0] #sum interval, sum epoch, count interval, count epoch
            datasets_loss_history[dataset_name] = []
        update_interval = 100 #how often loss is reported and stored
        count = 0
        for i in range(total_batches):
            count += 1
            # Select next dataset for batch selection:
            current_dataset_name = sampled_dataset_names_for_epoch[i]
            batch_X, batch_y = next(loader_iters[current_dataset_name])
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predicted = model.forward(batch_X, current_dataset_name)
            # Reshape for loss computation: [batch_size * seq_len, num_labels]
            loss = loss_function(predicted.view(-1, predicted.size(-1)), batch_y.flatten())
            allsets_epoch_loss += loss.item()

            dataset_losses[current_dataset_name][0] += loss.item() #sum interval - gets reset every update interval
            dataset_losses[current_dataset_name][1] += 1 #count interval - gets reset every update interval
            dataset_losses[current_dataset_name][2] += loss.item() # sum epoch
            dataset_losses[current_dataset_name][3] += 1 #count epoch

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            if count % update_interval == 0: 
                #get the average loss per dataset head every {update interval} batches to ensure things are working and shows the history
                clear_output(wait=True)
                print(f"Epoch {epoch + 1}\n-------------------------------")
                print(f"Average loss per dataset head over the last {update_interval} batches ({count} batches into epoch):")
                for dataset_name, (interval_sum, interval_count, _, _) in dataset_losses.items():
                    avg_dataset_loss = interval_sum / interval_count
                    print(f"  {dataset_name}: {avg_dataset_loss:.2f}")
                    temp_arr = copy.deepcopy(datasets_loss_history[dataset_name])
                    temp_arr.reverse()
                    print(f"        Previous {dataset_name} average losses, (every {update_interval} batches) most recent to earliest: {temp_arr}") 
                    datasets_loss_history[dataset_name].append(round(avg_dataset_loss, 2)) #append after such that current sum isn't displayed in history
                    print(f"        For reference, {dataset_name} has {DATASET_LABEL_SIZES[dataset_name]-1} possible labels.") #excludes <pad>
                    dataset_losses[dataset_name][0] = 0 #reset interval_sum 
                    dataset_losses[dataset_name][1] = 0 #reset interval_count
        allsets_avg_loss = allsets_epoch_loss / total_batches
        print(f"Average overall loss after epoch {epoch + 1}: {allsets_avg_loss:.2f}")
        for dataset_name, (_, _, epoch_sum, epoch_count) in dataset_losses.items():
            currset_avg_loss = epoch_sum / epoch_count
            print(f"Average {dataset_name} loss after epoch {epoch + 1}: {currset_avg_loss:.2f}")
    return model



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


#load_dotenv()  # load environment variables if needed
mlflow.set_tracking_uri("https://dagshub.com/PLtier/NLP-Cyber-NER.mlflow")



if __name__ == '__main__' and DEDUPLICATION == False:
    #yes it's a little scuffed this is here, but initially I forgot we had to do deduplication, so I'll just keep it, in case
    #it's ever of interest. I can't be bothered to merge it more neatly with the deduplication experiment, since here the model
    #is trained only once simply followed by a for loop for evaluations
    #hasn't been changed since deduplication version was done, so might not work optimally anymore

    cyner_train_X, cyner_train_y, cyner_dev_X, cyner_dev_y, cyner_test_X, cyner_test_y, cyner_idx2word, cyner_idx2label, cyner_max_len = \
    transform_dataset(
        cyner_train_data, cyner_dev_data, cyner_test_data,
    )

    attackner_train_X, attackner_train_y, attackner_dev_X, attackner_dev_y, attackner_test_X, attackner_test_y, attackner_idx2word, attackner_idx2label, attackner_max_len = \
    transform_dataset(
        attackner_train_data, attackner_dev_data, attackner_test_data
    )

    aptner_train_X, aptner_train_y, aptner_dev_X, aptner_dev_y, aptner_test_X, aptner_test_y, aptner_idx2word, aptner_idx2label, aptner_max_len = \
    transform_dataset(
        aptner_train_data, aptner_dev_data, aptner_test_data
    )

    dnrti_train_X, dnrti_train_y, dnrti_dev_X, dnrti_dev_y, dnrti_test_X, dnrti_test_y, dnrti_idx2word, dnrti_idx2label, dnrti_max_len = \
    transform_dataset(
        dnrti_train_data, dnrti_dev_data, dnrti_test_data
    )

    #Dataloaders:
    TRAIN_LOADERS = {
        "dnrti": DataLoader(TensorDataset(dnrti_train_X, dnrti_train_y), BATCH_SIZE, shuffle=True),
        "aptner": DataLoader(TensorDataset(aptner_train_X, aptner_train_y), BATCH_SIZE, shuffle=True),
        "attackner": DataLoader(TensorDataset(attackner_train_X, attackner_train_y), BATCH_SIZE, shuffle=True),
        "cyner": DataLoader(TensorDataset(cyner_train_X, cyner_train_y), BATCH_SIZE, shuffle=True),
    }

    DATASETS = ["dnrti", "aptner", "attackner", "cyner"]

    VOCABULARY_SIZES = [
        len(dnrti_idx2word), 
        len(aptner_idx2word), 
        len(attackner_idx2word), 
        len(cyner_idx2word)
    ]
    LABEL_SET_SIZES = [
        len(dnrti_idx2label), 
        len(aptner_idx2label), 
        len(attackner_idx2label), 
        len(cyner_idx2label)
    ]

    #function is probably unnecessary - could easily be set up manually, since VOCABULARY_SIZES, and LABEL_SET_SIZES is already
    #set up manually anyway
    DATASET_LABEL_SIZES, DATASET_VOCAB_SIZES = buildtokenmodeldicts(DATASETS, VOCABULARY_SIZES, LABEL_SET_SIZES)

    total_batches, batch_sampling_probs = getBatchSamplingProbs(TRAIN_LOADERS)

    #load_dotenv()  # load environment variables if needed
    mlflow.set_tracking_uri("file:///tmp/mlruns_nlpproject2025")
    mlflow.set_experiment("Never been more over than it is right now")


    with mlflow.start_run(run_name="TokenModel_sepEMB_tiedLSTM_train"):
        # Log hyperparameters
        hyperparams = {
            "BATCH_SIZE": BATCH_SIZE,
            "DIM_EMBEDDING": DIM_EMBEDDING,
            "LSTM_HIDDEN": LSTM_HIDDEN,
            "DROPOUT1": DROPOUT1,
            "DROPOUT2": DROPOUT2,
            "LEARNING_RATE": LEARNING_RATE,
            "EPOCHS": EPOCHS,
            "CLIPPING": CLIPPING, #We don't need clipping, but using same hyperparams as original model.
        }
        mlflow.log_params(hyperparams)

        # Instantiate and move model to device.
        model = TokenModel_sepEMB_tiedLSTM(DATASET_VOCAB_SIZES, DATASET_LABEL_SIZES)
        model = model.to(device)

        # Train the model.
        model = train_tokenmodel(model, TRAIN_LOADERS, total_batches, batch_sampling_probs, EPOCHS, device, DATASETS, CLIPPING)

        # Everything that may be needed:
        dev_data_dict = {
            "dnrti": (dnrti_dev_X, dnrti_dev_y, dnrti_idx2word, dnrti_idx2label, dnrti_dev_data),
            "aptner": (aptner_dev_X, aptner_dev_y, aptner_idx2word, aptner_idx2label, aptner_dev_data),
            "attackner": (attackner_dev_X, attackner_dev_y, attackner_idx2word, attackner_idx2label, attackner_dev_data),
            "cyner": (cyner_dev_X, cyner_dev_y, cyner_idx2word, cyner_idx2label, cyner_dev_data),
        }

        model.eval()

        # Nested runs for evlauation:
        for dataset_name in DATASETS:
            dev_X, dev_y, idx2word, idx2label, dev_data = dev_data_dict[dataset_name] #dev_y not currently used - this should change
            #once we've settled on hyperparams, such that we can include it in training.
            with mlflow.start_run(run_name=f"TokenModel_sepEMB_tiedLSTM_eval_{dataset_name}", nested=True):
                # Put the model in evaluation mode.
                dev_tokens, gold_labels = list(zip(*dev_data))
                dev_loader = DataLoader(TensorDataset(dev_X, dev_y), batch_size=BATCH_SIZE) #dev_y not currently used 
                allpredictions = []
                with torch.no_grad():
                    for batch_x, _ in dev_loader:
                        batch_x = batch_x.to(device)
                        batch_probas_for_y = model.forward(batch_x, dataset_name)
                        allpredictions.append(batch_probas_for_y.cpu())
                probas_dev = torch.cat(allpredictions, dim=0)
                pred_labels_idx_dev = torch.argmax(probas_dev, 2)
                predicted_tags = preds_to_tags(idx2label, gold_labels, pred_labels_idx_dev)
                metrics = span_f1(gold_labels, predicted_tags)
                mlflow.log_metrics(metrics)


        # Clean up.
        gc.collect()
        torch.cuda.empty_cache()





if __name__ == '__main__' and DEDUPLICATION == True:

        cyner_train_data, _ = remove_leakage(cyner_train_data, DATASETS_DATAPACK_DEV["cyner"])
        attackner_train_data, _ = remove_leakage(attackner_train_data, DATASETS_DATAPACK_DEV["attackner"])
        aptner_train_data, _ = remove_leakage(aptner_train_data, DATASETS_DATAPACK_DEV["aptner"])
        dnrti_train_data, _ = remove_leakage(dnrti_train_data, DATASETS_DATAPACK_DEV["dnrti"])

        cyner_train_X, cyner_train_y, cyner_dev_X, cyner_dev_y, cyner_test_X, cyner_test_y, cyner_idx2word, cyner_idx2label, cyner_max_len = \
        transform_dataset(
            cyner_train_data, cyner_dev_data, cyner_test_data,
        )

        attackner_train_X, attackner_train_y, attackner_dev_X, attackner_dev_y, attackner_test_X, attackner_test_y, attackner_idx2word, attackner_idx2label, attackner_max_len = \
        transform_dataset(
            attackner_train_data, attackner_dev_data, attackner_test_data
        )

        aptner_train_X, aptner_train_y, aptner_dev_X, aptner_dev_y, aptner_test_X, aptner_test_y, aptner_idx2word, aptner_idx2label, aptner_max_len = \
        transform_dataset(
            aptner_train_data, aptner_dev_data, aptner_test_data
        )

        dnrti_train_X, dnrti_train_y, dnrti_dev_X, dnrti_dev_y, dnrti_test_X, dnrti_test_y, dnrti_idx2word, dnrti_idx2label, dnrti_max_len = \
        transform_dataset(
            dnrti_train_data, dnrti_dev_data, dnrti_test_data
        )

        #Dataloaders:
        TRAIN_LOADERS = {
            "dnrti": DataLoader(TensorDataset(dnrti_train_X, dnrti_train_y), BATCH_SIZE, shuffle=True),
            "aptner": DataLoader(TensorDataset(aptner_train_X, aptner_train_y), BATCH_SIZE, shuffle=True),
            "attackner": DataLoader(TensorDataset(attackner_train_X, attackner_train_y), BATCH_SIZE, shuffle=True),
            "cyner": DataLoader(TensorDataset(cyner_train_X, cyner_train_y), BATCH_SIZE, shuffle=True),
        }

        DATASETS = ["dnrti", "aptner", "attackner", "cyner"]

        VOCABULARY_SIZES = [
            len(dnrti_idx2word), 
            len(aptner_idx2word), 
            len(attackner_idx2word), 
            len(cyner_idx2word)
        ]
        LABEL_SET_SIZES = [
            len(dnrti_idx2label), 
            len(aptner_idx2label), 
            len(attackner_idx2label), 
            len(cyner_idx2label)
        ]

        #function is probably unnecessary - could easily be set up manually, since VOCABULARY_SIZES, and LABEL_SET_SIZES is already
        #set up manually anyway
        DATASET_LABEL_SIZES, DATASET_VOCAB_SIZES = buildtokenmodeldicts(DATASETS, VOCABULARY_SIZES, LABEL_SET_SIZES)

        total_batches, batch_sampling_probs = getBatchSamplingProbs(TRAIN_LOADERS)

        mlflow.set_experiment(f"train-TokenModel_sepEMB_sepLSTM")
        with mlflow.start_run(run_name=f"train-TokenModel_sepEMB_sepLSTM"):
            # Log hyperparameters
            hyperparams = {
                "BATCH_SIZE": BATCH_SIZE,
                "DIM_EMBEDDING": DIM_EMBEDDING,
                "LSTM_HIDDEN": LSTM_HIDDEN,
                "DROPOUT1": DROPOUT1,
                "DROPOUT2": DROPOUT2,
                "LEARNING_RATE": LEARNING_RATE,
                "EPOCHS": EPOCHS,
                "CLIPPING": CLIPPING, #We don't need clipping, but using same hyperparams as original model.
            }
            mlflow.log_params(hyperparams)

            # Instantiate and move model to device.
            model = TokenModel_sepEMB_tiedLSTM(DATASET_VOCAB_SIZES, DATASET_LABEL_SIZES)
            model = model.to(device)

            # Train the model.
            model = train_tokenmodel(model, TRAIN_LOADERS, total_batches, batch_sampling_probs, EPOCHS, device, DATASETS, CLIPPING)

            # Everything that may be needed:
            dev_data_dict = {
                "dnrti": (dnrti_dev_X, dnrti_dev_y, dnrti_idx2word, dnrti_idx2label, dnrti_dev_data),
                "aptner": (aptner_dev_X, aptner_dev_y, aptner_idx2word, aptner_idx2label, aptner_dev_data),
                "attackner": (attackner_dev_X, attackner_dev_y, attackner_idx2word, attackner_idx2label, attackner_dev_data),
                "cyner": (cyner_dev_X, cyner_dev_y, cyner_idx2word, cyner_idx2label, cyner_dev_data),
            }

            model.eval()

            for intended_eval_dataset in DATASETS_DATAPACK_DEV.keys():
                with mlflow.start_run(run_name=f"TokenModel_sepEMB_sepLSTM_eval_{intended_eval_dataset}", nested=True):
                    dev_X, dev_y, idx2word, idx2label, dev_data = dev_data_dict[intended_eval_dataset] #dev_y not currently used - this should change
                    #once we've settled on hyperparams, such that we can include it in training.
                    
                    dev_tokens, gold_labels = list(zip(*dev_data))
                    dev_loader = DataLoader(TensorDataset(dev_X, dev_y), batch_size=BATCH_SIZE) #dev_y not currently used 
                    allpredictions = []
                    with torch.no_grad():
                        for batch_x, _ in dev_loader:
                            batch_x = batch_x.to(device)
                            batch_probas_for_y = model.forward(batch_x, intended_eval_dataset)
                            allpredictions.append(batch_probas_for_y.cpu())
                    probas_dev = torch.cat(allpredictions, dim=0)
                    pred_labels_idx_dev = torch.argmax(probas_dev, 2)
                    predicted_tags = preds_to_tags(idx2label, gold_labels, pred_labels_idx_dev)
                    metrics = span_f1(gold_labels, predicted_tags)

                    mlflow.log_metrics(metrics)
                    mlflow.log_param("TAG_SET_SIZE", len(idx2label)-1)
                    tag_set = idx2label[1:]
                    with open("ner_tags.json", "w") as f:
                        json.dump(tag_set, f)
                    mlflow.log_artifact("ner_tags.json") #ignore <pad>


            # Clean up.
            gc.collect()
            torch.cuda.empty_cache()
