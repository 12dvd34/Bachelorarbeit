import random
from utils import Dataset
# splits data into training- and testset


class DatasetGenerator:
    def __init__(self, data_provider, data_processor):
        self.data_provider = data_provider
        self.data_processor = data_processor
        self.dataset = Dataset.Dataset()


# produces same output when same input is given
class DeterministicGenerator(DatasetGenerator):
    def __init__(self, data_provider, data_processor):
        super().__init__(data_provider, data_processor)

    def generate_dataset(self, relative_testset_size=0.1):
        count = relative_testset_size
        for index, word in enumerate(self.data_provider.words):
            count += relative_testset_size
            input = self.data_processor.process_word(word)
            output = self.data_processor.process_label(self.data_provider.labels[index])
            if count >= 1:
                count = relative_testset_size
                self.dataset.testset.append((input, output))
            else:
                self.dataset.trainset.append((input, output))
        return self.dataset


# only here for historic reasons, not used anywhere
class RandomGenerator(DatasetGenerator):
    def __init__(self, data_provider, data_processor):
        super().__init__(data_provider, data_processor)

    def generate_dataset(self, relative_testset_size=0.1):
        for index, word in enumerate(self.data_provider.words):
            input = self.data_processor.process_word(word)
            output = self.data_processor.process_label(self.data_provider.labels[index])
            if random.random() < relative_testset_size:
                self.dataset.testset.append((input, output))
            else:
                self.dataset.trainset.append((input, output))
        return self.dataset
