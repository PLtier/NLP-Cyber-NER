from os import truncate
from pathlib import Path

import jsonlines
import torch

aptner_labels = set(
    [
        "B-TIME",
        "I-TIME",
        "E-TIME",
        "S-LOC",
        "B-SECTEAM",
        "E-SECTEAM",
        "I-SECTEAM",
        "S-SECTEAM",
        "S-TOOL",
        "B-IDTY",
        "E-IDTY",
        "S-MAL",
        "B-APT",
        "E-APT",
        "B-TOOL",
        "E-TOOL",
        "S-VULNAME",
        "S-VULID",
        "S-IDTY",
        "B-LOC",
        "E-LOC",
        "I-TOOL",
        "I-IDTY",
        "S-ENCR",
        "S-FILE",
        "S-SHA2",
        "S-URL",
        "S-IP",
        "S-APT",
        "PROT",
        "B-ACT",
        "E-ACT",
        "S-MD5",
        "I-ACT",
        "B-FILE",
        "E-FILE",
        "S-DOM",
        "B-MAL",
        "E-MAL",
        "S-OS",
        "S-TIME",
        "S-PROT",
        "S-ACT",
        "B-OS",
        "E-OS",
        "I-FILE",
        "S-SHA1",
        "B-URL",
        "E-URL",
        "B-IP",
        "E-IP",
        "S-M",
        "I-MAL",
        "B-SHA2",
        "E-SHA2",
        "B-VULNAME",
        "I-VULNAME",
        "E-VULNAME",
        "I-URL",
        "I-LOC",
        "I-APT",
        "I-OS",
        "B-PROT",
        "I-PROT",
        "E-PROT",
        "S-EMAIL",
        "B-VULID",
        "B-EMAIL",
        "E-EMAIL",
        "B-ENCR",
        "E-ENCR",
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
                tok = line.split(" ")
                if len(tok) == 1:
                    if tok[0] == "O":
                        continue
                    else:
                        f_out.write(f"{tok[0]} O\n")
                elif len(tok) == 2 and tok[1] not in aptner_labels:
                    f_out.write(f"{tok[0]} O\n")
                elif len(tok) >= 3:
                    # fuzzy cleaning: just pick the first token, and label it as O
                    # it's very rare.
                    f_out.write(f"{tok[0]} O\n")
                else:
                    f_out.write(line + "\n")
            else:
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
                tok = line.split(" ")
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


def read_iob2_file(path, sep="\t", word_index=1, tag_index=2):
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
            if line[0] == "#":
                continue  # skip comments
            tok = line.split(sep)

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


def read_cyner(path, sep="\t", word_index=0, tag_index=1):
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
            if line[0] == "#":
                continue  # skip comments
            tok = line.split(sep)

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
        print(i)
        line = line.strip()

        if line:
            tok = line.split(sep)

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


def read_dnrti(path: Path, sep=" ", word_index=0, tag_index=1):
    """Read dnrti NER dataset."""
    data = []
    current_words = []
    current_tags = []

    for i, line in enumerate(open(path, encoding="utf-8")):
        # print(i)
        # print(line)
        line = line.strip()

        if line:
            tok = line.split(sep)

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
        data_X = torch.zeros(instances, features, dtype=int)
        data_y = torch.zeros(instances, features, dtype=int)
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
        data_X = torch.zeros(instances, n_max_feats, dtype=int)
        data_y = torch.zeros(instances, n_max_feats, dtype=int)
        for i, sentence_tags in enumerate(data):
            truncated_sentence = sentence_tags[0][:n_max_feats]
            for j, word in enumerate(truncated_sentence):
                data_X[i, j] = self.vocab_words.getIdx(word=word, add=False)
                data_y[i, j] = self.vocab_tags.getIdx(word=sentence_tags[1][j], add=False)
        return data_X, data_y


def prepare_output_file(
    transformer: Preprocess,
    data: list,
    pred_labels: torch.Tensor,
    input_file: str,
    output_file: str,
):
    global_labels = []
    for (_, placeholder), labels_idxs in zip(data, pred_labels):
        labels = []

        for i in range(len(placeholder)):
            labels.append(transformer.vocab_tags.idx2word[labels_idxs[i]])
        global_labels += labels

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
                    words = line.split("\t")
                    words[2] = global_labels[i]
                    i += 1

                    new_line = "\t".join(words)
                    f_out.write(new_line)
            else:
                f_out.write("\n")
    assert i == len(global_labels)


def transform_dataset(train_data, dev_data, test_data):
    """
    Uses bag of words to create a matrix of words and their tags.
    """
    transformer = Preprocess()
    max_len = max([len(x[0]) for x in train_data])

    train_X, train_y, idx2word, idx2label = transformer.build_vocab(
        train_data, len(train_data), max_len
    )

    dev_X, dev_y = transformer.transform_prep_data(dev_data, len(dev_data), max_len)

    test_X, test_y = transformer.transform_prep_data(test_data, len(test_data), max_len)
    # here, the second variable doesn't hold true labels, as this is a test set. We need only to know the length of the sentences.
    return (
        train_X,
        train_y,
        dev_X,
        dev_y,
        test_X,
        test_y,
        idx2word,
        idx2label,
    )
