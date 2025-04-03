from os import truncate
from pathlib import Path

import jsonlines
from numpy.random import f
import torch

aptner_labels = set(
    [
        "B-ACT",
        "B-APT",
        "B-EMAIL",
        "B-ENCR",
        "B-FILE",
        "B-IDTY",
        "B-IP",
        "B-LOC",
        "B-MAL",
        "B-OS",
        "B-PROT",
        "B-SECTEAM",
        "B-SHA2",
        "B-TIME",
        "B-TOOL",
        "B-URL",
        "B-VULID",
        "B-VULNAME",
        "E-ACT",
        "E-APT",
        "E-EMAIL",
        "E-ENCR",
        "E-FILE",
        "E-IDTY",
        "E-IP",
        "E-LOC",
        "E-MAL",
        "E-OS",
        "E-PROT",
        "E-SECTEAM",
        "E-SHA2",
        "E-TIME",
        "E-TOOL",
        "E-URL",
        "E-VULNAME",
        "I-ACT",
        "I-APT",
        "I-FILE",
        "I-IDTY",
        "I-LOC",
        "I-MAL",
        "I-OS",
        "I-PROT",
        "I-SECTEAM",
        "I-TIME",
        "I-TOOL",
        "I-URL",
        "I-VULNAME",
        "S-ACT",
        "S-APT",
        "S-DOM",
        "S-EMAIL",
        "S-ENCR",
        "S-FILE",
        "S-IDTY",
        "S-IP",
        "S-LOC",
        "S-MAL",
        "S-MD5",
        "S-OS",
        "S-PROT",
        "S-SECTEAM",
        "S-SHA1",
        "S-SHA2",
        "S-TIME",
        "S-TOOL",
        "S-URL",
        "S-VULID",
        "S-VULNAME",
    ]
)


def clean_aptner(path: Path) -> None:
    """
    If there is only one token in the sentence, it is assigned the label O. The function saves cleaned data.
    """
    with (
        open(path, "r", encoding="utf-8") as f,
        open(path.with_suffix(".cleaned"), "w", encoding="utf-8") as f_out,
    ):
        for line in f:
            line = line.strip()
            if line:
                tok = line.split()
                if len(tok) == 1:
                    if tok[0] == "O":
                        continue
                    else:
                        f_out.write(f"{tok[0]} O\n")
                elif len(tok) == 2 and (tok[1] not in aptner_labels or "-" not in tok[1]):
                    f_out.write(f"{tok[0]} O\n")
                elif len(tok) >= 3:
                    # fuzzy cleaning: just pick the first token, and label it as O
                    # it's very rare.
                    f_out.write(f"{tok[0]} O\n")
                else:
                    f_out.write(line + "\n")
            else:
                f_out.write("\n")


def unify_labels_aptner(path: Path) -> None:
    """
    All E- labels are converted to I- labels.
    All S- labels are converted to B- labels.
    APT, SECTEAM -> Organization (respectively, B- and I-)
    OS -> System (respectively, B- and I-)
    VULNAME -> Vulnerability (respectively, B- and I-)
    MAL -> Malware (respectively, B- and I-)
    """

    with (
        open(path, "r", encoding="utf-8") as f,
        open(path.with_suffix(".unified"), "w", encoding="utf-8") as f_out,
    ):
        for line in f:
            line = line.strip()
            if line:
                print(line)
                tok = line.split()
                assert len(tok) == 2
                new_tag = tok[1]
                if tok[1] != "O":
                    prefix, label = tok[1].split("-")

                    if label == "APT" or label == "SECTEAM":
                        label = "Organization"
                    elif label == "OS":
                        label = "System"
                    elif label == "VULNAME":
                        label = "Vulnerability"
                    elif label == "MAL":
                        label = "Malware"
                    else:
                        label = "O"
                        f_out.write(f"{tok[0]} O\n")
                        continue

                    if prefix == "E":
                        prefix = "I"
                    elif prefix == "S":
                        prefix = "B"

                    new_tag = f"{prefix}-{label}"
                f_out.write(f"{tok[0]} {new_tag}\n")
            else:
                f_out.write("\n")


def unify_labels_dnrti(path: Path) -> None:
    """
    HackOrg, SecTeam -> Organization (respectively, B- and I-)
    Tool -> System (respectively, B- and I-)
    Way -> Vulnerability (respectively, B- and I-)
    SamFile -> Malware (respectively, B- and I-)
    conll to conll
    """

    with (
        open(path, "r", encoding="utf-8") as f,
        open(path.with_suffix(".unified"), "w", encoding="utf-8") as f_out,
    ):
        for line in f:
            line = line.strip()
            if line:
                tok = line.split()
                assert len(tok) == 2
                new_tag = tok[1]
                if tok[1] != "O":
                    prefix, label = tok[1].split("-")

                    if label == "HackOrg" or label == "SecTeam":
                        label = "Organization"
                    elif label == "Tool":
                        label = "System"
                    elif label == "Way":
                        label = "Vulnerability"
                    elif label == "SamFile":
                        label = "Malware"
                    else:
                        label = "O"
                        f_out.write(f"{tok[0]} O\n")
                        continue

                    new_tag = f"{prefix}-{label}"
                f_out.write(f"{tok[0]} {new_tag}\n")
            else:
                f_out.write("\n")


def unify_labels_attackner(path: Path) -> None:
    """
    THREAT_ACTOR, GENERAL_IDENTITY -> Organization (respectively, B- and I-)
    INFRASTRUCTURE, GENERAL_TOOL, ATTACK_TOOL -> Infrastructure (respectively, B- and I-)
    VULNERABILITY -> Vulnerability (respectively, B- and I-)
    MALWARE -> Malware (respectively, B- and I-)
    Outputs a conll format!
    """
    with (
        jsonlines.open(path) as reader,
        open(path.with_suffix(".unified"), "w", encoding="utf-8") as f_out,
    ):
        for obj in reader:
            tags = obj["tags"]
            tokens = obj["tokens"]
            n = len(tokens)
            for i in range(n):
                current_tag = tags[i]
                token = tokens[i]
                if token == " ":
                    # TODO: this is kind of cleaning part, if there is time, I would put it in a separate function
                    continue
                if current_tag != "O":
                    prefix, label = current_tag.split("-")

                    if label == "THREAT_ACTOR" or label == "GENERAL_IDENTITY":
                        label = "Organization"
                    elif (
                        label == "INFRASTRUCTURE"
                        or label == "GENERAL_TOOL"
                        or label == "ATTACK_TOOL"
                    ):
                        label = "System"
                    elif label == "VULNERABILITY":
                        label = "Vulnerability"
                    elif label == "MALWARE":
                        label = "Malware"
                    else:
                        label = "O"
                        f_out.write(f"{token} O\n")
                        continue
                    current_tag = f"{prefix}-{label}"
                f_out.write(f"{token} {current_tag}\n")
            f_out.write("\n")


def clean_dnrti(path: Path) -> None:
    """
    If there is only one token in the sentence, it is assigned the label O. The function saves cleaned data.
    """
    with (
        open(path, "r", encoding="utf-8") as f,
        open(path.with_suffix(".cleaned"), "w", encoding="utf-8") as f_out,
    ):
        for line in f:
            line = line.strip()
            if line:
                tok = line.split()
                if len(tok) == 1:
                    if tok[0] == "O":
                        continue
                    else:
                        f_out.write(f"{tok[0]} O\n")
                elif len(tok) >= 3:
                    # fuzzy cleaning: just pick the first token, and label it as O
                    # it's very rare.
                    f_out.write(f"{tok[0]} O\n")
                else:
                    f_out.write(line + "\n")
            else:
                f_out.write("\n")


def read_iob2_file(path, word_index=0, tag_index=1):
    """
    read in conll file with no comments

    :param path: path to read from
    :returns: list with sequences of words and labels for each sentence
    """
    data = []
    current_words = []
    current_tags = []

    for line in open(path, encoding="utf-8"):
        line = line.strip()

        if line:
            # if line[0] == "#" and len(line.split()) == 1:
            # continue  # skip comments
            # TODO: introduce comments, such that hashtags don't break
            tok = line.split()

            current_words.append(tok[word_index])
            current_tags.append(tok[tag_index])
        else:
            if current_words:  # skip empty lines
                data.append((current_words, current_tags))
            current_words = []
            current_tags = []

    # check for last one
    if current_tags != []:
        data.append((current_words, current_tags))
    return data


def read_cyner(path, word_index=0, tag_index=1):
    """
    read in conll file

    :param path: path to read from
    :returns: list with sequences of words and labels for each sentence
    """
    data = []
    current_words = []
    current_tags = []

    for line in open(path, encoding="utf-8"):
        line = line.strip()

        if line:
            tok = line.split()

            current_words.append(tok[word_index])
            tag = tok[tag_index]
            current_tags.append(tag)
        else:
            if current_words:  # skip empty lines
                data.append((current_words, current_tags))
            current_words = []
            current_tags = []

    # check for last one
    if current_tags != []:
        data.append((current_words, current_tags))
    return data


def read_aptner(path, sep=" ", word_index=0, tag_index=1):
    """
    read in conll file

    :param path: path to read from
    :returns: list with sequences of words and labels for each sentence
    """
    data = []
    current_words = []
    current_tags = []

    for i, line in enumerate(open(path, encoding="utf-8")):
        # print(i)
        line = line.strip()

        if line:
            tok = line.split()

            current_words.append(tok[word_index])
            tag = tok[tag_index]
            current_tags.append(tag)
        else:
            if current_words:  # skip empty lines
                data.append((current_words, current_tags))
            current_words = []
            current_tags = []

    # check for last one
    if current_tags != []:
        data.append((current_words, current_tags))
    return data


def read_attackner(path):
    """
    read in conll file

    :param path: path to read from
    :returns: list with sequences of words and labels for each sentence
    """
    data = []

    with jsonlines.open(path) as reader:
        for obj in reader:
            tags = obj["tags"]
            tokens = obj["tokens"]
            data.append((tokens, tags))
    return data


def read_dnrti(path: Path, word_index=0, tag_index=1):
    """Read dnrti NER dataset."""
    data = []
    current_words = []
    current_tags = []

    for i, line in enumerate(open(path, encoding="utf-8")):
        # print(i)
        # print(line)
        line = line.strip()

        if line:
            tok = line.split()

            current_words.append(tok[word_index])
            tag = tok[tag_index]
            current_tags.append(tag)
        else:
            if current_words:  # skip empty lines
                data.append((current_words, current_tags))
            current_words = []
            current_tags = []

    # check for last one
    if current_tags != []:
        data.append((current_words, current_tags))
    return data


class Vocab:
    def __init__(self, pad_unk="<PAD>"):
        """
        A convenience class that can help store a vocabulary
        and retrieve indices for inputs.
        """
        self.pad_unk = pad_unk
        self.word2idx = {self.pad_unk: 0}
        self.idx2word = [self.pad_unk]

    def getIdx(self, word, add=False):
        if word not in self.word2idx:
            if add:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)
            else:
                return self.word2idx[self.pad_unk]
        return self.word2idx[word]

    def getWord(self, idx):
        return self.idx2word[idx]


# Your implementation goes here:


class Preprocess:
    """
    data: the dataset from which we get the matrix used by a Neural network (instances + their tags)
    instances: number of instances in the dataset, needed for dimension of matrix
    features: the number of features/columns of the matrix
    """

    def __init__(self):
        self.vocab_words = Vocab()
        self.vocab_tags = Vocab()

    def build_vocab(self, data, instances, features):
        data_X = torch.zeros(instances, features, dtype=torch.int)
        data_y = torch.zeros(instances, features, dtype=torch.int)
        for i, sentence_tags in enumerate(data):
            for j, word in enumerate(sentence_tags[0]):
                data_X[i, j] = self.vocab_words.getIdx(word=word, add=True)
                data_y[i, j] = self.vocab_tags.getIdx(word=sentence_tags[1][j], add=True)

        # returns the list of unique words in the list from the attributes of the Vocab()
        idx2word_train = self.vocab_words.idx2word
        # returns the list of unique tags in the list from the attributes of the Vocab()
        idx2label_train = self.vocab_tags.idx2word
        # only returned in the builder function, because they are reused for dev data in transform_prep_data()
        return data_X, data_y, idx2word_train, idx2label_train

    def transform_prep_data(self, data, instances, n_max_feats: int):
        # to be used only on dev data
        data_X = torch.zeros(instances, n_max_feats, dtype=torch.int)
        data_y = torch.zeros(instances, n_max_feats, dtype=torch.int)
        for i, sentence_tags in enumerate(data):
            truncated_sentence = sentence_tags[0][:n_max_feats]
            for j, word in enumerate(truncated_sentence):
                data_X[i, j] = self.vocab_words.getIdx(word=word, add=False)
                data_y[i, j] = self.vocab_tags.getIdx(word=sentence_tags[1][j], add=False)
        return data_X, data_y


def get_labels(
    gt_data: list[list[tuple[str]]],
):
    """
    Justr retursn labels, instead of tags
    """
    return [labels for _, labels in gt_data]


def preds_to_tags(idx2word, pred_labels: torch.Tensor, gt_data: list[list[tuple[str]]]):
    """
    Converts the predicted labels to tags.
    """
    global_labels = []
    for (_, placeholder), labels_idxs in zip(gt_data, pred_labels):
        labels = []

        for i in range(len(placeholder)):
            labels.append(idx2word[labels_idxs[i]])
        global_labels.append(labels)
    return global_labels


def prepare_output_file(
    transformer: Preprocess,
    gt_data: list[list[tuple[str]]],
    pred_labels: torch.Tensor,
    input_file: str,
    output_file: str,
):
    global_labels = preds_to_tags(transformer, pred_labels, gt_data)

    with (
        open(output_file, mode="w", encoding="utf-8") as f_out,
        open(input_file, mode="r", encoding="utf-8") as f_in,
    ):
        i = 0
        for line in f_in.readlines():
            if line.strip():
                if line[0] == "#":
                    f_out.write(line)
                else:
                    words = line.split()
                    words[2] = global_labels[i]
                    i += 1

                    new_line = "\t".join(words)
                    f_out.write(new_line)
            else:
                f_out.write("\n")
    assert i == len(global_labels)


def transform_dataset(train_data, dev_data, test_data, return_dev_test_labels=False):
    """
    Uses bag of words to create a matrix of words and their tags.
    Instead of returning a matrix of numbers for dev/test, it just returns the (human-readable) tags.
    """
    transformer = Preprocess()
    max_len = max([len(x[0]) for x in train_data])

    train_X, train_y, idx2word, idx2label = transformer.build_vocab(
        train_data, len(train_data), max_len
    )

    dev_X, dev_y = transformer.transform_prep_data(dev_data, len(dev_data), max_len)

    test_X, test_y = transformer.transform_prep_data(test_data, len(test_data), max_len)
    # here, the second variable doesn't hold true labels, as this is a test set. We need only to know the length of the sentences.
    if return_dev_test_labels:
        dev_y = get_labels(dev_data)
        test_y = get_labels(test_data)
    return (
        train_X,
        train_y,
        dev_X,
        dev_y,
        test_X,
        test_y,
        idx2word,
        idx2label,
        max_len,
    )
