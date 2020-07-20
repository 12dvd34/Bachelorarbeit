import torch
import json
from io import StringIO
# use this instead of DataProvider


def load_file(file, sample_size=0):
    data = json.load(StringIO(file.read()))
    if sample_size == 0:
        return data
    else:
        return data[:sample_size]


def load_weights(file):
    text = file.read()
    io = StringIO(text)
    weights_list = json.load(io)
    weights_tensor = torch.zeros(len(weights_list), len(weights_list[0]))
    for row, rows in enumerate(weights_list):
        for col, num in enumerate(rows):
            weights_tensor[row][col] = float(num)
    return weights_tensor


def generate_token_mapping(words):
    assert len(words) > 0, "no words found"
    mapping = {}
    for word in words:
        for token in word:
            if token not in mapping:
                mapping[token] = len(mapping)
    return mapping


def generate_label_mapping(labels):
    assert len(labels) > 0, "no labels found"
    mapping = {}
    for label in labels:
        if label not in mapping:
            mapping[label] = len(mapping)
    return mapping


class DataProviderLight:
    def __init__(self, file_words, file_labels, file_weights=None, sample_size=0):
        self.words = load_file(file_words, sample_size)
        self.labels = load_file(file_labels, sample_size)
        if file_weights is not None:
            self.weights = load_weights(file_weights)
        self.token_mapping = generate_token_mapping(self.words)
        self.label_mapping = generate_label_mapping(self.labels)

    def get_max_word_length(self):
        max_length = 0
        for word in self.words:
            if len(word) > max_length:
                max_length = len(word)
        return max_length

    # returns all tokens that appear in an equation together with the given token
    def find_shared_tokens(self, token):
        assert len(self.words) > 0, "no words found"
        shared_tokens = []
        for word in self.words:
            if token in word:
                for tkn in word:
                    if tkn is not token and tkn not in shared_tokens:
                        shared_tokens.append(tkn)
        return shared_tokens

    # generates an one-hot vector encoding the given token
    def generate_token_one_hot(self, token):
        assert len(self.token_mapping) > 0, "no token mapping found"
        token_vector = torch.zeros(len(self.token_mapping))
        token_vector[self.token_mapping[token]] = 1
        return token_vector
