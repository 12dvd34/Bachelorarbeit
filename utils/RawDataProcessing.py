import torch
# generated embeddings for raw data


class RawDataProcessor:
    def __init__(self, data_provider):
        self.data_provider = data_provider

    # generates an one-hot vector encoding the given token
    def generate_token_one_hot(self, token):
        assert len(self.data_provider.token_mapping) > 0, "no token mapping found"
        token_vector = torch.zeros(len(self.data_provider.token_mapping))
        token_vector[self.data_provider.token_mapping[token]] = 1
        return token_vector

    # generates a 2D word vector with an one-hot vector for each token
    def generate_word_one_hot(self, word, padding=False):
        assert len(self.data_provider.token_mapping) > 0, "no token mapping found"
        word_length = len(word)
        if padding:
            word_length = self.data_provider.get_max_word_length()
        word_vector = torch.zeros(word_length, len(self.data_provider.token_mapping))
        for index_token, token in enumerate(word):
            word_vector[index_token][self.data_provider.token_mapping[token]] = 1
        return word_vector

    # generates an one-hot vector encoding the given label
    def generate_label_one_hot(self, label):
        assert len(self.data_provider.label_mapping) > 0, "no label mapping found"
        label_vector = torch.zeros(len(self.data_provider.label_mapping))
        label_vector[self.data_provider.label_mapping[label]] = 1
        return label_vector

    # returns the vector representation of the token with the given index
    def generate_token_vector(self, token_index):
        assert self.data_provider.weights is not None, "no weights found"
        token_vector = torch.zeros(self.data_provider.weights.size(0))
        for i in range(self.data_provider.weights.size(0)):
            token_vector[i] = self.data_provider.weights[i][token_index]
        return token_vector

    # returns a vector representing the mean of all the words token vectors
    def generate_word_vector(self, word):
        assert self.data_provider.weights is not None, "no weights found"
        assert len(self.data_provider.token_mapping) > 0, "no token mapping found"
        mean = torch.zeros(self.data_provider.weights.size(0))
        for token in word:
            mean += self.generate_token_vector(self.data_provider.token_mapping[token])
        mean /= len(word)
        return mean


# one-hots for use with RNN
class OneHotProcessor(RawDataProcessor):
    def __init__(self, data_provider):
        super().__init__(data_provider)

    def process_word(self, word, padding=False):
        return super().generate_word_one_hot(word, padding)

    def process_label(self, label):
        return torch.tensor([self.data_provider.label_mapping[label]], dtype=torch.long)


# one-hots for use with FFN
class FlatOneHotProcessor(RawDataProcessor):
    def __init__(self, data_provider):
        super().__init__(data_provider)

    def process_word(self, word):
        return super().generate_word_one_hot(word, True).flatten()

    def process_label(self, label):
        return torch.tensor([self.data_provider.label_mapping[label]], dtype=torch.long)


# vectors for use with FFN or KNN
class VectorProcessor(RawDataProcessor):
    def __init__(self, data_provider):
        super().__init__(data_provider)

    def process_word(self, word):
        return super().generate_word_vector(word)

    def process_label(self, label):
        return torch.tensor([self.data_provider.label_mapping[label]], dtype=torch.long)


# vectors for use with RNN
class TokenVectorProcessor(RawDataProcessor):
    def __init__(self, data_provider, padding=False):
        super().__init__(data_provider)
        self.padding = padding

    def process_word(self, word, padding=False):
        if padding or self.padding:
            word_vector = torch.zeros((self.data_provider.get_max_word_length(), self.data_provider.weights.size(0)), dtype=torch.float)
        else:
            word_vector = torch.zeros((len(word), self.data_provider.weights.size(0)), dtype=torch.float)
        for token_index, token in enumerate(word):
            token_vector = super().generate_token_vector(self.data_provider.token_mapping[token])
            for num_index in range(len(token_vector)):
                word_vector[token_index][num_index] = token_vector[num_index]
        return word_vector

    def process_label(self, label):
        return torch.tensor([self.data_provider.label_mapping[label]], dtype=torch.long)


# same as VectorProcessor but with precalculated embeddings for faster loading
class PregeneratedProcessor(RawDataProcessor):
    def __init__(self, data_provider):
        super().__init__(data_provider)

    def process_word(self, word):
        return torch.tensor(word)

    def process_label(self, label):
        return torch.tensor([self.data_provider.label_mapping[label]], dtype=torch.long)
