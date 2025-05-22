# The same as train.py except it uses the same tags tha the tokenmodel uses, which is close to the original set of tags for the
# datasets, except some trivial ones have been droppped.

import gc
import json
import mlflow
import torch
import random 
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from compute_fn_ratios import false_negative_ratio, MAPPED_LABELS


from nlp_cyber_ner.config import DATA_DIR, PROCESSED_DATA_DIR, TOKENPROCESSED_DATA_DIR, load_dotenv
from nlp_cyber_ner.dataset import (
    Preprocess,
    preds_to_tags,
    read_iob2_file,
    remove_leakage,
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
CLIPPING = 5.0 # - didn't seem like it was needed from testing, but used in the original model (train.py) so reusing here


#Enable Deduplication:
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


class TaggerModel(torch.nn.Module):
    def __init__(self, nwords, ntags):
        super().__init__()
        self.embed = nn.Embedding(nwords, DIM_EMBEDDING)
        self.drop1 = nn.Dropout(p=0.5)
        self.lstm = nn.LSTM(DIM_EMBEDDING, LSTM_HIDDEN, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc = nn.Linear(LSTM_HIDDEN * 2, ntags)

    def forward(self, input_data):
        word_vectors = self.embed(input_data)
        regular1 = self.drop1(word_vectors)
        output, hidden = self.lstm(regular1)
        regular2 = self.drop2(output)

        predictions = self.fc(regular2)
        return predictions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

mlflow.set_tracking_uri("https://dagshub.com/PLtier/NLP-Cyber-NER.mlflow")

def train(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    idx2word: list[str],
    idx2label: list[str],
    input_size: int,
) -> TaggerModel:
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, BATCH_SIZE)  # drop_last=True
    n_batches = len(train_loader)

    model = TaggerModel(len(idx2word), len(idx2label))
    model = model.to(device)  # run on cuda if possible
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")

    # creating the batches

    for epoch in range(EPOCHS):
        model.train()
        # reset the gradient
        print(f"Epoch {epoch + 1}\n-------------------------------")
        loss_sum = 0

        # loop over batches
        # types for convenience
        batch_X: torch.Tensor
        batch_y: torch.Tensor
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # run on cuda if possible

            optimizer.zero_grad()

            predicted_values = model.forward(batch_X)

            # Cross entropy request (predictions, classes) shape for predictions, and (classes) for batch_y

            # calculate loss
            loss = loss_function(
                predicted_values.view(batch_X.shape[0] * input_size, -1), batch_y.flatten()
            )  # TODO: Input
            loss_sum += loss.item()  # avg later

            # update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIPPING)
            optimizer.step()

        print(f"Average loss after epoch {epoch + 1}: {loss_sum / n_batches}")

    # set to evaluation mode
    model.eval()

    gc.collect()
    torch.cuda.empty_cache()

    return model




def test_pass(
    model: TaggerModel,
    dev_X: torch.Tensor,
    return_labels_idx: bool = True,
) -> torch.Tensor:
    """
    Performs forward pass on the model and returns the predictions.
    if return_labels is True, returns the labels, instead of probabilities.
    """
    all_predictions = []

    dev_dataset = TensorDataset(dev_X)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

    with torch.no_grad():
        for (batch_dev,) in dev_loader:
            batch_dev = batch_dev.to(device)  # run on cuda if possible
            batch_probas_dev = model.forward(batch_dev)
            all_predictions.append(batch_probas_dev.cpu())

    # eval using Span_F1

    # to minimise memory usage, we delete the model and dev_X, even though it takes performance hit
    del dev_X
    del model
    gc.collect()
    torch.cuda.empty_cache()

    probas_dev = torch.cat(all_predictions, dim=0)
    if not return_labels_idx:
        return probas_dev
    else:
        labels_idx_dev = torch.argmax(probas_dev, 2)
        return labels_idx_dev


def evaluate(
    gt_labels: tuple,
    pred_labels: list,
):
    metrics = span_f1(gt_labels, pred_labels)

    return metrics


train_packs = [
    ("cyner", cyner_train_data),
    ("aptner", aptner_train_data),
    ("attackner", attackner_train_data),
    ("dnrti", dnrti_train_data),
]

dev_packs = [
    ("cyner", cyner_dev_data),
    ("aptner", aptner_dev_data),
    ("attackner", attackner_dev_data),
    ("dnrti", dnrti_dev_data),
]


for i in range(len(train_packs)):
    
    train_data, dev_data = train_packs[i][1], dev_packs[i][1]

    train_data, _ = remove_leakage(train_data, dev_data)
    transformer = Preprocess()

    max_len = max([len(x[0]) for x in train_data])

    # input_size = max_len
    train_X, train_y, idx2word, idx2label = transformer.build_vocab(
        train_data, len(train_data), max_len
    )

    name: str = f"train-tokenreferencemodel-eval-{dev_packs[i][0]}"
    print(f"train {name}")
    model = train(train_X, train_y, idx2word, idx2label, input_size=max_len)

    dev_X, _ = transformer.transform_prep_data(dev_data, len(dev_data), max_len)
    dev_tokens, dev_labels = list(zip(*dev_data))

    # train_X, train_y, dev_X, dev_y, test_X, test_y, idx2word, idx2label, max_len = train_pack
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=name):
        assert isinstance(dev_labels, tuple), "Dev y is not a list!"

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

        pred_labels_idx_dev = test_pass(model, dev_X, return_labels_idx=True)

        labels_dev = preds_to_tags(idx2label, dev_labels, pred_labels_idx_dev)

        metrics = evaluate(dev_labels, labels_dev)

        mlflow.log_params(hyperparams)
        mlflow.log_metrics(metrics)
        mlflow.log_params(
            {
                "train_size": train_X.shape[0],
                "train_input_size": train_X.shape[1],
                "dev_size": dev_X.shape[0],
                "dev_input_size": max(len(x[0]) for x in dev_data),
            }
        )

        dataset_name  = dev_packs[i][0]         
        mapped_set = MAPPED_LABELS[dataset_name]
        fn_ratio  = false_negative_ratio(dev_labels, labels_dev, mapped_set)
        mlflow.log_metric("fn_ratio_mapped", fn_ratio)

        mlflow.log_param("TAG_SET_SIZE", len(idx2label)-1)
        tag_set = idx2label[1:]
        with open("ner_tags.json", "w") as f:
            json.dump(tag_set, f)
        mlflow.log_artifact("ner_tags.json") #ignore <pad>
        
