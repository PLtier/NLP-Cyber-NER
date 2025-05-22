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
import os

from nlp_cyber_ner.config import PROCESSED_DATA_DIR, TOKENPROCESSED_DATA_DIR, load_dotenv

from nlp_cyber_ner.dataset import read_iob2_file, preds_to_tags, remove_leakage, Vocab
from nlp_cyber_ner.span_f1 import span_f1

# Hyperparameters
BATCH_SIZE = 32
DIM_EMBEDDING = 100
LSTM_HIDDEN = 100
DROPOUT1 = 0.5
DROPOUT2 = 0.5
LEARNING_RATE = 0.001
EPOCHS = 15
CLIPPING = 5.0  # - didn't seem like it was needed from testing, but used in the original model (train.py) so reusing here


# Enable Deduplication:
DEDUPLICATION = True


# Setting up datasets:
# CYNER: (not changes for tokenmodel so loaded just from processed directory)
cyner_path = PROCESSED_DATA_DIR / "cyner"
cyner_train_path = cyner_path / "train.unified"
cyner_dev_path = cyner_path / "valid.unified"
cyner_test_path = cyner_path / "test.unified"
cyner_train_data = read_iob2_file(cyner_train_path)
cyner_dev_data = read_iob2_file(cyner_dev_path)
cyner_test_data = read_iob2_file(cyner_test_path)

# ATTACKNER:
attackner_path = TOKENPROCESSED_DATA_DIR / "attacker"
attackner_train_path = attackner_path / "train.tokenready"
attackner_dev_path = attackner_path / "dev.tokenready"
attackner_test_path = attackner_path / "test.tokenready"
attackner_train_data = read_iob2_file(attackner_train_path, word_index=0, tag_index=1)
attackner_dev_data = read_iob2_file(attackner_dev_path, word_index=0, tag_index=1)
attackner_test_data = read_iob2_file(attackner_test_path, word_index=0, tag_index=1)

# APTNER:
aptner_path = TOKENPROCESSED_DATA_DIR / "APTNer"
aptner_train_path = aptner_path / "train.tokenready"
aptner_dev_path = aptner_path / "valid.tokenready"
aptner_test_path = aptner_path / "test.tokenready"
aptner_train_data = read_iob2_file(aptner_train_path)
aptner_dev_data = read_iob2_file(aptner_dev_path)
aptner_test_data = read_iob2_file(aptner_test_path)

# DNRTI:
dnrti_path = TOKENPROCESSED_DATA_DIR / "DNRTI"
dnrti_train_path = dnrti_path / "train.tokenready"
dnrti_dev_path = dnrti_path / "valid.tokenready"
dnrti_test_path = dnrti_path / "test.tokenready"
dnrti_train_data = read_iob2_file(dnrti_train_path, word_index=0, tag_index=1)
dnrti_dev_data = read_iob2_file(dnrti_dev_path, word_index=0, tag_index=1)
dnrti_test_data = read_iob2_file(dnrti_test_path, word_index=0, tag_index=1)


# names of the current datasets
DATASETS = ["dnrti", "aptner", "attackner", "cyner"]

DATASETS_DATAPACK_TRAIN = {  # uses the data structures resulting from the red_iob2_file function - i.e. list of tuples of lists.
    "dnrti": dnrti_train_data,
    "aptner": aptner_train_data,
    "attackner": attackner_train_data,
    "cyner": cyner_train_data,
}

DATASETS_DATAPACK_DEV = {
    "dnrti": dnrti_dev_data,
    "aptner": aptner_dev_data,
    "attackner": attackner_dev_data,
    "cyner": cyner_dev_data,
}

DATASETS_DATAPACK_TEST = {
    "dnrti": dnrti_test_data,
    "aptner": aptner_test_data,
    "attackner": attackner_test_data,
    "cyner": cyner_test_data,
}


def build_vocab_commonvocab_tokenmodel(
    data: dict, instances: dict, n_features: dict, datasets: list
):
    """
    Similar to our build vocab funtion. Builds 4 token X tensors, and tag Y tensors. The 4 token X tensors, shares the same vocabulary object.
    The 4 tag Y tensors do not, as the heads of the model are still individualized.
    """

    commonvocab_words = Vocab()
    tag_vocabs = {}
    pytorch_tensors = {}
    for dataset_name in datasets:
        tag_vocabs[dataset_name] = Vocab()
        current_ds_tag_vocab = tag_vocabs[dataset_name]
        data_X = torch.zeros(instances[dataset_name], n_features[dataset_name], dtype=torch.long)
        data_y = torch.zeros(instances[dataset_name], n_features[dataset_name], dtype=torch.long)
        for i, sentence_tags in enumerate(data[dataset_name]):
            for j, word in enumerate(sentence_tags[0]):
                data_X[i, j] = commonvocab_words.getIdx(word=word, add=True)
                data_y[i, j] = current_ds_tag_vocab.getIdx(word=sentence_tags[1][j], add=True)
        pytorch_tensors[dataset_name] = (data_X, data_y)

    idx2word_train = commonvocab_words.idx2word

    return pytorch_tensors, idx2word_train, tag_vocabs, commonvocab_words


def transform_prep_data_commonvocabtokenmodel(
    data, instances, n_max_feats: int, common_vocab, tag_vocab
):  # for dev and test sets:
    """
    common vocab should be the vocab constructed based on all datasets for tokens.
    Tag_vocab is the dataset-specific vocab for tags/labels.
    """
    data_X = torch.zeros(instances, n_max_feats, dtype=torch.long)
    data_y = torch.zeros(instances, n_max_feats, dtype=torch.long)
    for i, sentence_tags in enumerate(data):
        truncated_sentence = sentence_tags[0][:n_max_feats]
        for j, word in enumerate(truncated_sentence):
            data_X[i, j] = common_vocab.getIdx(word=word, add=False)
            data_y[i, j] = tag_vocab.getIdx(word=sentence_tags[1][j], add=False)
    return data_X, data_y


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


class TokenModel_tiedEMB_tiedLSTM(nn.Module):
    def __init__(self, datasets_vocab_size, dataset_label_map):
        super().__init__()
        self.embedding = nn.Embedding(datasets_vocab_size, DIM_EMBEDDING)

        self.drop1 = nn.Dropout(p=DROPOUT1)

        self.rnn = nn.LSTM(DIM_EMBEDDING, LSTM_HIDDEN, batch_first=True, bidirectional=True)

        self.drop2 = nn.Dropout(p=DROPOUT2)

        self.heads = nn.ModuleDict()
        for dataset_name, ntags in dataset_label_map.items():
            self.heads[dataset_name] = nn.Linear(LSTM_HIDDEN * 2, ntags)

    def forward(self, input_data, dataset_name):
        word_vectors = self.embedding(input_data)
        regular1 = self.drop1(word_vectors)
        output, hidden = self.rnn(regular1)
        regular2 = self.drop2(output)

        if dataset_name not in self.heads:
            raise ValueError(
                f"Dataset id '{dataset_name}' not recognized in heads. "
                f"Available heads: {list(self.heads.keys())}"
            )
        predictions = self.heads[dataset_name](regular2)  # [batch_size, max_len, num_labels]
        return predictions


def train_tokenmodel(
    model, train_loaders, total_batches, sampling_probs, epochs, device, datasets, max_grad_norm
) -> TokenModel_tiedEMB_tiedLSTM:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    torch.manual_seed(0)

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch + 1}\n-------------------------------")
        allsets_epoch_loss = 0.0

        loader_iters = {
            dataset_name: cycle(loader) for dataset_name, loader in train_loaders.items()
        }

        sampled_dataset_names_for_epoch = np.random.choice(
            datasets, size=total_batches, p=sampling_probs
        )

        dataset_losses = {}
        datasets_loss_history = {}
        for dataset_name in datasets:
            dataset_losses[dataset_name] = [
                0.0,
                0.0,
                0,
                0,
            ]  # sum interval, sum epoch, count interval, count epoch
            datasets_loss_history[dataset_name] = []
        update_interval = 100  # how often loss is reported and stored
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

            dataset_losses[current_dataset_name][0] += (
                loss.item()
            )  # sum interval - gets reset every update interval
            dataset_losses[current_dataset_name][1] += (
                1  # count interval - gets reset every update interval
            )
            dataset_losses[current_dataset_name][2] += loss.item()  # sum epoch
            dataset_losses[current_dataset_name][3] += 1  # count epoch

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            if count % update_interval == 0:
                # get the average loss per dataset head every {update interval} batches to ensure things are working and shows the history
                clear_output(wait=True)
                print(f"Epoch {epoch + 1}\n-------------------------------")
                print(
                    f"Average loss per dataset head over the last {update_interval} batches ({count} batches into epoch):"
                )
                for dataset_name, (interval_sum, interval_count, _, _) in dataset_losses.items():
                    avg_dataset_loss = interval_sum / interval_count
                    print(f"  {dataset_name}: {avg_dataset_loss:.2f}")
                    temp_arr = copy.deepcopy(datasets_loss_history[dataset_name])
                    temp_arr.reverse()
                    print(
                        f"        Previous {dataset_name} average losses, (every {update_interval} batches) most recent to earliest: {temp_arr}"
                    )
                    datasets_loss_history[dataset_name].append(
                        round(avg_dataset_loss, 2)
                    )  # append after such that current sum isn't displayed in history
                    print(
                        f"        For reference, {dataset_name} has {DATASET_LABEL_SIZES[dataset_name] - 1} possible labels."
                    )  # excludes <pad>
                    dataset_losses[dataset_name][0] = 0  # reset interval_sum
                    dataset_losses[dataset_name][1] = 0  # reset interval_count
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

load_dotenv()
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri is not None:
    mlflow.set_tracking_uri(tracking_uri)


if __name__ == "__main__" and DEDUPLICATION == False:
    # yes it's a little scuffed this is here, and it could probably easily be merged with the deduplication experiment more neatly,
    # but initially I forgot we had to do deduplication, so I'll just keep it like this in case it's ever of interest
    # hasn't been changed since deduplication version was done, so might not work optimally anymore

    # load_dotenv()  # load environment variables if needed
    mlflow.set_experiment("TokenModel_tiedEMB_tiedLSTM")

    max_len_dict = {}
    n_instances_dict = {}
    for dataset_name in DATASETS:
        max_len_dict[dataset_name] = max(
            [len(x[0]) for x in DATASETS_DATAPACK_TRAIN[dataset_name]]
        )
        n_instances_dict[dataset_name] = len(DATASETS_DATAPACK_TRAIN[dataset_name])

    TENSOR_DATAPACK, common_idx2word, DATASET_TAG_VOCABS, commonvocab_words = (
        build_vocab_commonvocab_tokenmodel(
            DATASETS_DATAPACK_TRAIN, n_instances_dict, max_len_dict, DATASETS
        )
    )

    TRAIN_LOADERS = {}
    for dataset_name in DATASETS:
        TRAIN_LOADERS[dataset_name] = DataLoader(
            TensorDataset(*TENSOR_DATAPACK[dataset_name]), batch_size=BATCH_SIZE, shuffle=True
        )

    DATASET_LABEL_SIZES = {}
    for dataset_name in DATASETS:
        DATASET_LABEL_SIZES[dataset_name] = len(DATASET_TAG_VOCABS[dataset_name].idx2word)

    common_vocab_size = len(common_idx2word)

    EVALREADY_DATAPACK_DEV = {}
    for dataset_name in DATASETS:
        EVALREADY_DATAPACK_DEV[dataset_name] = transform_prep_data_commonvocabtokenmodel(
            DATASETS_DATAPACK_DEV[dataset_name],
            len(DATASETS_DATAPACK_DEV[dataset_name]),
            max_len_dict[dataset_name],
            commonvocab_words,
            DATASET_TAG_VOCABS[dataset_name],
        ) + (
            commonvocab_words.idx2word,
            DATASET_TAG_VOCABS[dataset_name].idx2word,
            DATASETS_DATAPACK_DEV[dataset_name],
        )

    total_num_batches, batch_sampling_probs = getBatchSamplingProbs(TRAIN_LOADERS)

    with mlflow.start_run(run_name="TokenModel_tiedEMB_tiedLSTM_train"):
        # Log hyperparameters
        hyperparams = {
            "BATCH_SIZE": BATCH_SIZE,
            "DIM_EMBEDDING": DIM_EMBEDDING,
            "LSTM_HIDDEN": LSTM_HIDDEN,
            "DROPOUT1": DROPOUT1,
            "DROPOUT2": DROPOUT2,
            "LEARNING_RATE": LEARNING_RATE,
            "EPOCHS": EPOCHS,
            "CLIPPING": CLIPPING,  # We don't need clipping, but using same hyperparams as original model.
        }
        mlflow.log_params(hyperparams)

        # Instantiate and move model to device.
        model = TokenModel_tiedEMB_tiedLSTM(len(common_idx2word), DATASET_LABEL_SIZES)
        model = model.to(device)

        # Train the model.
        model = train_tokenmodel(
            model,
            TRAIN_LOADERS,
            total_num_batches,
            batch_sampling_probs,
            EPOCHS,
            device,
            DATASETS,
            CLIPPING,
        )

        model.eval()

        # Nested runs for evlauation:
        for dataset_name in DATASETS:
            dev_X, dev_y, idx2word, idx2label, dev_data = EVALREADY_DATAPACK_DEV[
                dataset_name
            ]  # dev_y not currently used - this should change
            # once we've settled on hyperparams, such that we can include it in training.
            with mlflow.start_run(
                run_name=f"TokenModel_tiedEMB_tiedLSTM_eval_{dataset_name}", nested=True
            ):
                # Put the model in evaluation mode.
                dev_tokens, gold_labels = list(zip(*dev_data))
                dev_loader = DataLoader(
                    TensorDataset(dev_X, dev_y), batch_size=BATCH_SIZE
                )  # dev_y not currently used
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


if __name__ == "__main__" and DEDUPLICATION == True:
    # Even with deduplication, we only get 4 total experiments.

    for intended_eval_dataset in DATASETS:
        DEDUPLICATED_DATAPACK_TRAIN = {}
        for dataset_name in DATASETS:
            # deduplicate all training sets from intended evaluation dataset
            DEDUPLICATED_DATAPACK_TRAIN[dataset_name], _ = remove_leakage(
                DATASETS_DATAPACK_TRAIN[dataset_name], DATASETS_DATAPACK_DEV[intended_eval_dataset]
            )

        max_len_dict = {}
        n_instances_dict = {}
        for dataset_name in DATASETS:
            max_len_dict[dataset_name] = max(
                [len(x[0]) for x in DEDUPLICATED_DATAPACK_TRAIN[dataset_name]]
            )
            n_instances_dict[dataset_name] = len(DEDUPLICATED_DATAPACK_TRAIN[dataset_name])

        TENSOR_DATAPACK, common_idx2word, DATASET_TAG_VOCABS, commonvocab_words = (
            build_vocab_commonvocab_tokenmodel(
                DEDUPLICATED_DATAPACK_TRAIN, n_instances_dict, max_len_dict, DATASETS
            )
        )

        TRAIN_LOADERS = {}
        for dataset_name in DATASETS:
            TRAIN_LOADERS[dataset_name] = DataLoader(
                TensorDataset(*TENSOR_DATAPACK[dataset_name]), batch_size=BATCH_SIZE, shuffle=True
            )

        DATASET_LABEL_SIZES = {}
        for dataset_name in DATASETS:
            DATASET_LABEL_SIZES[dataset_name] = len(DATASET_TAG_VOCABS[dataset_name].idx2word)

        common_vocab_size = len(common_idx2word)

        EVALREADY_DATAPACK_DEV = {}
        for dataset_name in DATASETS:
            EVALREADY_DATAPACK_DEV[dataset_name] = transform_prep_data_commonvocabtokenmodel(
                DATASETS_DATAPACK_DEV[dataset_name],
                len(DATASETS_DATAPACK_DEV[dataset_name]),
                max_len_dict[dataset_name],
                commonvocab_words,
                DATASET_TAG_VOCABS[dataset_name],
            ) + (
                commonvocab_words.idx2word,
                DATASET_TAG_VOCABS[dataset_name].idx2word,
                DATASETS_DATAPACK_DEV[dataset_name],
            )

        total_num_batches, batch_sampling_probs = getBatchSamplingProbs(TRAIN_LOADERS)

        mlflow.set_experiment(f"train-TokenModel_tiedEMB_tiedLSTM-eval-{intended_eval_dataset}")
        with mlflow.start_run(
            run_name=f"train-TokenModel_tiedEMB_tiedLSTM-eval-{intended_eval_dataset}"
        ):
            # Log hyperparameters
            hyperparams = {
                "BATCH_SIZE": BATCH_SIZE,
                "DIM_EMBEDDING": DIM_EMBEDDING,
                "LSTM_HIDDEN": LSTM_HIDDEN,
                "DROPOUT1": DROPOUT1,
                "DROPOUT2": DROPOUT2,
                "LEARNING_RATE": LEARNING_RATE,
                "EPOCHS": EPOCHS,
                "CLIPPING": CLIPPING,  # We don't need clipping, but using same hyperparams as original model.
            }
            mlflow.log_params(hyperparams)

            # Instantiate and move model to device.
            model = TokenModel_tiedEMB_tiedLSTM(len(common_idx2word), DATASET_LABEL_SIZES)
            model = model.to(device)

            # Train the model.
            model = train_tokenmodel(
                model,
                TRAIN_LOADERS,
                total_num_batches,
                batch_sampling_probs,
                EPOCHS,
                device,
                DATASETS,
                CLIPPING,
            )

            model.eval()

            dev_X, dev_y, idx2word, idx2label, dev_data = EVALREADY_DATAPACK_DEV[
                intended_eval_dataset
            ]  # dev_y not currently used - this should change
            # once we've settled on hyperparams, such that we can include it in training.

            dev_tokens, gold_labels = list(zip(*dev_data))
            dev_loader = DataLoader(
                TensorDataset(dev_X, dev_y), batch_size=BATCH_SIZE
            )  # dev_y not currently used
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
            mlflow.log_param("TAG_SET_SIZE", len(idx2label) - 1)
            tag_set = idx2label[1:]
            with open("ner_tags.json", "w") as f:
                json.dump(tag_set, f)
            mlflow.log_artifact("ner_tags.json")  # ignore <pad>

            # Clean up.
            gc.collect()
            torch.cuda.empty_cache()
