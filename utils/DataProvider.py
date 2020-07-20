import json
import torch
from io import StringIO
import random
# DON'T USE THIS
# it's horribly bloated but still used somewhere


class DataProvider:
    def __init__(self):
        self.words = []
        self.labels = []
        self.word_vectors = []
        self.token_mapping = {}
        self.label_mapping = {}
        self.weights = None
        self.trainset = []
        self.testset = []

    # reads the file to a list (words) of lists (characters)
    def load_tokens(self, file, sample_size=0):
        text = file.read()
        io = StringIO(text)
        words = json.load(io)
        if sample_size == 0:
            sample_size = len(words)
        sample = words[:sample_size]
        self.words = sample
        return sample

    # only for consistence, same as load_words
    def load_labels(self, file, sample_size=0):
        text = file.read()
        io = StringIO(text)
        labels = json.load(io)
        if sample_size == 0:
            sample_size = len(labels)
        sample = labels[:sample_size]
        self.labels = sample
        return sample

    # only for consistence, same as load_words
    def load_word_vectors(self, file, sample_size=0):
        word_vectors = json.load(StringIO(file.read()))
        if sample_size == 0:
            sample_size = len(word_vectors)
        sample = word_vectors[:sample_size]
        self.word_vectors = sample
        return sample

    # reads the weights generated by the Word2Vec network
    def load_weights(self, file):
        text = file.read()
        io = StringIO(text)
        weights_list = json.load(io)
        weights_tensor = torch.zeros(len(weights_list), len(weights_list[0]))
        for row, rows in enumerate(weights_list):
            for col, num in enumerate(rows):
                weights_tensor[row][col] = float(num)
        self.weights = weights_tensor
        return weights_tensor

    # inserts the weights as returned by get_weights() in Word2Vec
    def set_weights(self, weights_list):
        weights_tensor = torch.zeros(len(weights_list), len(weights_list[0]))
        for row, rows in enumerate(weights_list):
            for col, num in enumerate(rows):
                weights_tensor[row][col] = float(num)
        self.weights = weights_tensor

    # maps each token to a number
    def generate_token_mapping(self):
        assert len(self.words) > 0, "no words found"
        mapping = {}
        for word in self.words:
            for token in word:
                if token not in mapping:
                    mapping[token] = len(mapping)
        self.token_mapping = mapping
        return mapping

    # maps each label to an number
    def generate_label_mapping(self):
        assert len(self.labels) > 0, "no labels found"
        mapping = {}
        for label in self.labels:
            if label not in mapping:
                mapping[label] = len(mapping)
        self.label_mapping = mapping
        return mapping

    # generates an one-hot vector encoding the given token
    def generate_token_one_hot(self, token):
        assert len(self.token_mapping) > 0, "no token mapping found"
        token_vector = torch.zeros(len(self.token_mapping))
        token_vector[self.token_mapping[token]] = 1
        return token_vector

    # generates an one-hot vector encoding the given label
    def generate_label_one_hot(self, label):
        assert len(self.label_mapping) > 0, "no label mapping found"
        label_vector = torch.zeros(len(self.label_mapping))
        label_vector[self.label_mapping[label]] = 1
        return label_vector

    # generates a word vector with an one-hot tensor for each token
    def generate_word_one_hot(self, word, compress=True):
        assert len(self.token_mapping) > 0, "no token mapping found"
        word_vector = torch.zeros(len(word), len(self.token_mapping))
        for index_token, token in enumerate(word):
            word_vector[index_token][self.token_mapping[token]] = 1
        if compress:
            word_vector = word_vector.view(len(word) * len(self.token_mapping))
        return word_vector

    # generates a word vector with an one-hot tensor for each token, filled up to the length of the longest word
    def generate_padded_word_one_hot(self, word):
        assert len(self.token_mapping) > 0, "no token mapping found"
        max_length = self.get_max_word_length()
        word_vector = torch.zeros(max_length * len(self.token_mapping))
        for index_token, token in enumerate(word):
            word_vector[index_token * len(self.token_mapping) + self.token_mapping[token]] = 1
        return word_vector

    # generates a word vector with a number for each token
    def generate_padded_word_number_tensor(self, word):
        word_tensor = torch.zeros(self.get_max_word_length(), dtype=torch.long)
        for index in range(len(word_tensor)):
            if index < len(word):
                word_tensor[index] = self.token_mapping[word[index]]
            else:
                word_tensor[index] = len(self.token_mapping)
        return word_tensor

    # generates a word vector with a number for each token, but without padding
    def generate_word_number_tensor(self, word):
        word_tensor = torch.zeros(len(word), dtype=torch.long)
        for index in range(len(word_tensor)):
            word_tensor[index] = self.token_mapping[word[index]]
        return word_tensor

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

    # returns the vector representation of the token with the given index
    def generate_token_vector(self, token_index):
        assert self.weights is not None, "no weights found"
        token_vector = torch.zeros(self.weights.size(0))
        for i in range(self.weights.size(0)):
            token_vector[i] = self.weights[i][token_index]
        return token_vector

    # returns a vector representing the mean of all the words token vectors
    def generate_word_vector(self, word):
        assert self.weights is not None, "no weights found"
        assert len(self.token_mapping) > 0, "no token mapping found"
        mean = torch.zeros(self.weights.size(0))
        for token in word:
            mean += self.generate_token_vector(self.token_mapping[token])
        mean /= len(word)
        return mean

    # returns a vector representing the mean of all the words with the given label
    def generate_label_vector(self, label):
        assert len(self.label_mapping) > 0, "no label mapping found"
        mean = torch.zeros(self.weights.size(0))
        count = 0
        for index, word in enumerate(self.words):
            if self.labels[index] == label:
                mean += self.generate_word_vector(word)
                count += 1
        mean /= count
        return mean

    # splits the data into training and test data
    def generate_datasets(self, prob_testset=0.1):
        assert len(self.words) > 0, "no words found"
        assert len(self.labels) > 0, "no labels found"
        trainset = []
        train_data = []
        train_labels = []
        testset = []
        test_data = []
        test_labels = []
        assert len(self.words) > 0, "no words found"
        for index, word in enumerate(self.words):
            if random.random() <= prob_testset:
                test_data.append(word)
                test_labels.append(self.labels[index])
            else:
                train_data.append(word)
                train_labels.append(self.labels[index])
        trainset.append(train_data)
        trainset.append(train_labels)
        self.trainset = trainset
        self.testset = testset
        testset.append(test_data)
        testset.append(test_labels)
        return trainset, testset

    # splits the data into batches of the given size
    def generate_batches(self, data, batch_size=10, dtype=torch.float):
        batched_data = []
        for i in range(len(data))[::batch_size]:
            l = torch.zeros(len(data[i:i+batch_size]), len(data[0]), dtype=dtype)
            for j, elem in enumerate(data[i:i+batch_size]):
                l[j] = elem
            batched_data.append(l)
        return batched_data

    # splits the data into batches, all elements of the batch will have the same size
    def generate_padded_batches(self, data, batch_size=10, dtype=torch.float):
        assert len(self.token_mapping) > 0, "no token mapping found"
        batched_data = []
        for i in range(len(data))[::batch_size]:
            max_length = 0
            # get length of longest word in batch
            for j, elem in enumerate(data[i:i+batch_size]):
                if max_length < len(elem):
                    max_length = len(elem)
            # fill up all words to max length
            for _, elem in enumerate(data[i:i+batch_size]):
                while len(elem) < max_length:
                    elem.append(torch.zeros(len(self.token_mapping), dtype=dtype))
            # create batches
            batch = []
            for token_index in range(max_length):
                vec = torch.zeros(len(data[i:i+batch_size]), len(self.token_mapping))
                for word_index, word in enumerate(data[i:i+batch_size]):
                    for num_index, num in enumerate(word[token_index]):
                        vec[word_index][num_index] = num
                batch.append(vec)
            batched_data.append(batch)

    def get_max_word_length(self):
        max_length = 0
        for word in self.words:
            if len(word) > max_length:
                max_length = len(word)
        return max_length