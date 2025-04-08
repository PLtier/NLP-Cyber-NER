import gc
from pathlib import Path

from loguru import logger
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import typer

from nlp_cyber_ner.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

# from nlp_cyber_ner.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR
from nlp_cyber_ner.dataset import (
    Preprocess,
    clean_aptner,
    clean_dnrti,
    get_labels,
    preds_to_tags,
    read_aptner,
    read_attackner,
    read_cyner,
    read_dnrti,
    read_iob2_file,
    transform_dataset,
    unify_labels_aptner,
)
from nlp_cyber_ner.span_f1 import span_f1

# cyner
cyner_path = RAW_DATA_DIR / "cyner"
cyner_train_path = cyner_path / "train.txt"
cyner_dev_path = cyner_path / "valid.txt"
cyner_test_path = cyner_path / "test.txt"
cyner_train_data = read_cyner(cyner_train_path)
cyner_dev_data = read_cyner(cyner_dev_path)
cyner_test_data = read_cyner(cyner_test_path)


# aptner
aptner_path = PROCESSED_DATA_DIR / "APTNer"
aptner_train_path = aptner_path / "APTNERtrain.unified"
aptner_dev_path = aptner_path / "APTNERdev.unified"
aptner_test_path = aptner_path / "APTNERtest.unified"
aptner_train_data = read_aptner(aptner_train_path)
aptner_dev_data = read_aptner(aptner_dev_path)
aptner_test_data = read_aptner(aptner_test_path)
end_labels = {
    "B-Organization",
    "O",
    "I-Malware",
    "B-System",
    "I-Vulnerability",
    "I-Organization",
    "I-System",
    "B-Vulnerability",
    "B-Malware",
}
A = set(tag for _, tags in aptner_train_data for tag in tags)
B = set(tag for _, tags in aptner_dev_data for tag in tags)
C = set(tag for _, tags in aptner_test_data for tag in tags)
assert A == B == C == end_labels, "The labels in the train, dev and test sets are not the same."

# attackner
attackner_path = PROCESSED_DATA_DIR / "attackner"
attackner_train_path = attackner_path / "train.unified"
attackner_dev_path = attackner_path / "dev.unified"
attackner_test_path = attackner_path / "test.unified"
attackner_train = read_iob2_file(attackner_train_path, word_index=0, tag_index=1)
attackner_dev = read_iob2_file(attackner_dev_path, word_index=0, tag_index=1)
attackner_test = read_iob2_file(attackner_test_path, word_index=0, tag_index=1)

A = set(tag for _, tags in attackner_train for tag in tags)
B = set(tag for _, tags in attackner_dev for tag in tags)
C = set(tag for _, tags in attackner_test for tag in tags)
assert A == B == C == end_labels, "The labels in the train, dev and test sets are not the same."

dnrti_path = PROCESSED_DATA_DIR / "DNRTI"
dnrti_train_path = dnrti_path / "train.unified"
dnrti_dev_path = dnrti_path / "valid.unified"
dnrti_test_path = dnrti_path / "test.unified"

dnrti_train = read_iob2_file(dnrti_train_path, word_index=0, tag_index=1)
dnrti_dev = read_iob2_file(dnrti_dev_path, word_index=0, tag_index=1)
dnrti_test = read_iob2_file(dnrti_test_path, word_index=0, tag_index=1)
A = set(tag for _, tags in dnrti_train for tag in tags)
B = set(tag for _, tags in dnrti_dev for tag in tags)
C = set(tag for _, tags in dnrti_test for tag in tags)
assert A == B == C == end_labels, "The labels in the train, dev and test sets are not the same."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_eval(
    train_X,
    train_y,
    dev_X,
    dev,
    idx2word,
    idx2label,
    max_len,
    run: mlflow.ActiveRun = None,
) -> None:
    ### Transforms
    # !nvidia-smi
    # put already to gpu if having space:
    train_X, train_y = train_X.to(device), train_y.to(device)
    dev_X = dev_X.to(device)
    test_X = test_X.to(device)
    ### Batching

    # TODO: Maybe dtype would need to be changed!
    BATCH_SIZE = 32
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, BATCH_SIZE)  # drop_last=True
    n_batches = len(train_loader)
    ### Training

    torch.manual_seed(0)
    DIM_EMBEDDING = 100
    LSTM_HIDDEN = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    EPOCHS = 15

    class TaggerModel(torch.nn.Module):
        def __init__(self, nwords, ntags):
            super().__init__()
            # TODO Do Bidirectional LSTM
            self.embed = nn.Embedding(nwords, DIM_EMBEDDING)
            self.drop1 = nn.Dropout(p=0.2)
            self.rnn = nn.LSTM(DIM_EMBEDDING, LSTM_HIDDEN, batch_first=True, bidirectional=True)
            self.drop2 = nn.Dropout(p=0.3)
            self.fc = nn.Linear(LSTM_HIDDEN * 2, ntags)

        def forward(self, input_data):
            word_vectors = self.embed(input_data)
            regular1 = self.drop1(word_vectors)
            output, hidden = self.rnn(regular1)
            regular2 = self.drop2(output)

            predictions = self.fc(regular2)
            return predictions

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
            # TODO: if having memory issues comment .to(device)
            # from one of the previous cells, and uncomment that:
            # batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()

            predicted_values = model.forward(batch_X)

            # Cross entropy request (predictions, classes) shape for predictions, and (classes) for batch_y

            # calculate loss
            loss = loss_function(
                predicted_values.view(batch_X.shape[0] * max_len, -1), batch_y.flatten()
            )  # TODO: Last batch has 31 entries instead of 32 - we don't adjust much for that.
            loss_sum += loss.item()  # avg later

            # update
            loss.backward()
            optimizer.step()

        print(f"Average loss after epoch {epoch + 1}: {loss_sum / n_batches}")

    # set to evaluation mode
    model.eval()

    # eval using Span_F1
    predictions_dev = model.forward(dev_X)
    print(predictions_dev.shape)
    # gives probabilities for each tag (dim=18) for each word/feature (dim=159) for each sentence(dim=2000)
    # we want to classify each word for the part-of-speech with highest probability
    labels_dev = torch.argmax(predictions_dev, 2)
    print(labels_dev.shape)
    gt_labels = get_labels(dev)
    labels_dev = preds_to_tags(idx2word, labels_dev, dev)
    metrics = span_f1(gt_labels, labels_dev)

    run.log_param("epoch", epoch + 1)

    del predictions_dev
    del labels_dev
    gc.collect()
    torch.cuda.empty_cache()
