import gc

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nlp_cyber_ner.config import PROCESSED_DATA_DIR, load_dotenv
from nlp_cyber_ner.dataset import Preprocess, get_labels, preds_to_tags, read_iob2_file
from nlp_cyber_ner.span_f1 import span_f1

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

cyner_path = PROCESSED_DATA_DIR / "cyner"
cyner_train_path = cyner_path / "train.unified"
cyner_dev_path = cyner_path / "valid.unified"
cyner_test_path = cyner_path / "test.unified"

cyner_train_data = read_iob2_file(cyner_train_path)
cyner_dev_data = read_iob2_file(cyner_dev_path)
cyner_test_data = read_iob2_file(cyner_test_path)
A = set(tag for _, tags in cyner_train_data for tag in tags)
B = set(tag for _, tags in cyner_dev_data for tag in tags)
C = set(tag for _, tags in cyner_test_data for tag in tags)
assert A == B == C == end_labels, "The labels in the train, dev and test sets are not the same."

aptner_path = PROCESSED_DATA_DIR / "APTNer"
aptner_train_path = aptner_path / "APTNERtrain.unified"
aptner_dev_path = aptner_path / "APTNERdev.unified"
aptner_test_path = aptner_path / "APTNERtest.unified"
aptner_train_data = read_iob2_file(aptner_train_path)
aptner_dev_data = read_iob2_file(aptner_dev_path)
aptner_test_data = read_iob2_file(aptner_test_path)
A = set(tag for _, tags in aptner_train_data for tag in tags)
B = set(tag for _, tags in aptner_dev_data for tag in tags)
C = set(tag for _, tags in aptner_test_data for tag in tags)
assert A == B == C == end_labels, "The labels in the train, dev and test sets are not the same."

attackner_path = PROCESSED_DATA_DIR / "attackner"
attackner_train_path = attackner_path / "train.unified"
attackner_dev_path = attackner_path / "dev.unified"
attackner_test_path = attackner_path / "test.unified"
attackner_train_data = read_iob2_file(attackner_train_path, word_index=0, tag_index=1)
attackner_dev_data = read_iob2_file(attackner_dev_path, word_index=0, tag_index=1)
attackner_test_data = read_iob2_file(attackner_test_path, word_index=0, tag_index=1)
A = set(tag for _, tags in attackner_train_data for tag in tags)
B = set(tag for _, tags in attackner_dev_data for tag in tags)
C = set(tag for _, tags in attackner_test_data for tag in tags)
assert A == B == C == end_labels, (
    "The labels in the train_data, dev_data and test_data sets are not the same."
)

dnrti_path = PROCESSED_DATA_DIR / "dnrti"
dnrti_train_path = dnrti_path / "train.unified"
dnrti_dev_path = dnrti_path / "valid.unified"
dnrti_test_path = dnrti_path / "test.unified"

dnrti_train_data = read_iob2_file(dnrti_train_path, word_index=0, tag_index=1)
dnrti_dev_data = read_iob2_file(dnrti_dev_path, word_index=0, tag_index=1)
dnrti_test_data = read_iob2_file(dnrti_test_path, word_index=0, tag_index=1)
A = set(tag for _, tags in dnrti_train_data for tag in tags)
B = set(tag for _, tags in dnrti_dev_data for tag in tags)
C = set(tag for _, tags in dnrti_test_data for tag in tags)
assert A == B == C == end_labels, "The labels in the train, dev and test sets are not the same."

load_dotenv()
mlflow.set_tracking_uri("https://dagshub.com/PLtier/NLP-Cyber-NER.mlflow")

BATCH_SIZE = 32
DIM_EMBEDDING = 100
LSTM_HIDDEN = 100
LEARNING_RATE = 0.001
EPOCHS = 15
CLIPPING = 5.0
hyperparams = {
    "BATCH_SIZE": BATCH_SIZE,
    "DIM_EMBEDDING": DIM_EMBEDDING,
    "LSTM_HIDDEN": LSTM_HIDDEN,
    "LEARNING_RATE": LEARNING_RATE,
    "EPOCHS": EPOCHS,
    "CLIPPING": 5.0,
}


class TaggerModel(torch.nn.Module):
    def __init__(self, nwords, ntags):
        super().__init__()
        # TODO Do Bidirectional LSTM
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


def train(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    idx2word: list[str],
    idx2label: list[str],
    max_len: int,
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
                predicted_values.view(batch_X.shape[0] * max_len, -1), batch_y.flatten()
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


def evaluate(
    model: TaggerModel,
    dev_X: torch.Tensor,
    dev_labels: list[tuple[str]],
    idx2label: list[str],
) -> dict[str, float]:
    """
    Span F1 evaluation function.
    """
    all_predictions = []

    dev_dataset = TensorDataset(dev_X)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

    with torch.no_grad():
        for (batch_dev,) in dev_loader:
            batch_dev = batch_dev.to(device)  # run on cuda if possible
            predictions_dev = model.forward(batch_dev)

            all_predictions.append(predictions_dev.cpu())

    # eval using Span_F1
    predictions_dev = torch.cat(all_predictions, dim=0)
    print(predictions_dev.shape)
    labels_dev = torch.argmax(predictions_dev, 2)
    print(labels_dev.shape)

    labels_dev = preds_to_tags(idx2label, dev_labels, labels_dev)

    metrics = span_f1(dev_labels, labels_dev)

    del dev_X
    del predictions_dev
    del labels_dev
    del model
    gc.collect()
    torch.cuda.empty_cache()
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

for train_pack_name, train_data in train_packs:
    transformer = Preprocess()
    max_len = max([len(x[0]) for x in train_data])

    train_X, train_y, idx2word, idx2label = transformer.build_vocab(
        train_data, len(train_data), max_len
    )

    model = train(train_X, train_y, idx2word, idx2label, max_len)

    for dev_pack_name, dev_data in dev_packs:
        dev_X, _ = transformer.transform_prep_data(dev_data, len(dev_data), max_len)
        dev_y = get_labels(dev_data)

        name: str = f"train-{train_pack_name}-eval-{dev_pack_name}"

        # train_X, train_y, dev_X, dev_y, test_X, test_y, idx2word, idx2label, max_len = train_pack
        mlflow.set_experiment(name)
        with mlflow.start_run(run_name=name):
            assert isinstance(dev_y, list), "Dev y is not a list!"
            metrics = evaluate(model, dev_X, dev_y, idx2label)
            mlflow.log_params(hyperparams)
            mlflow.log_metrics(metrics)
