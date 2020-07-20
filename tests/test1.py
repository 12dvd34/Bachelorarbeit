import torch
from utils.RawDataProcessing import VectorProcessor
from utils.DataProviderLight import DataProviderLight
from utils.DatasetGeneration import DeterministicGenerator
from preprocessing.Word2Vec import Word2Vec
from classification.KNNClassification import KNNClassifier
from classification.ClassificationTest import ClassificationTest
# script for corresponding test case
# most test cases should be able to be executed without any further changes, if data is available

SAMPLE_SIZE = 10000
FEATURES = 50
BATCH_SIZE = 32
EPOCHS = [1, 5, 10, 20, 40]
# change device to "cpu" if cuda not available
DEVICE = "cuda"

# raw words and labels
file_words = open("../data/unique_equations.json")
file_labels = open("../data/unique_labels.json")
# pre calculated weight matrix
file_weights = open("../data/weights_0.json")
data_provider = DataProviderLight(file_words, file_labels, sample_size=SAMPLE_SIZE, file_weights=file_weights)
# embedding data, splitting up into train and test set
processor = VectorProcessor(data_provider)
generator = DeterministicGenerator(data_provider, processor)

for num_epochs in EPOCHS:
    # training the word2vec net
    word2vec = Word2Vec(data_provider, FEATURES, DEVICE)
    word2vec.train(num_epochs, BATCH_SIZE)
    # extracting weights and injecting them into the data provider
    data_provider.weights = torch.tensor(word2vec.get_weights())
    # generate dataset
    dataset = generator.generate_dataset()
    # train knn classifier
    classifier = KNNClassifier(data_provider, dataset, DEVICE)
    classifier.n_neighbours = 5
    classifier.train()
    # test the classifier
    test = ClassificationTest(dataset, classifier)
    print(str(num_epochs) + " Epochen: " + str(test.test()) + "% Präzision")