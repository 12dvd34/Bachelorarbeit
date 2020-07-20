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
FEATURES = [25, 50, 100, 200, 400]
BATCH_SIZE = 32
EPOCHS = 5
# change device to "cpu" if cuda not available
DEVICE = "cuda"

# raw words and labels
file_words = open("../data/unique_equations.json")
file_labels = open("../data/unique_labels.json")
# pre calculated weight matrix
file_weights = open("../data/weights_0.json")
data_provider = DataProviderLight(file_words, file_labels, sample_size=SAMPLE_SIZE, file_weights=file_weights)

for num_features in FEATURES:
    # training the word2vec net
    word2vec = Word2Vec(data_provider, num_features, DEVICE)
    word2vec.train(EPOCHS, BATCH_SIZE)
    # extracting weights and injecting them into the data provider
    data_provider.weights = torch.tensor(word2vec.get_weights())
    # generate dataset
    processor = VectorProcessor(data_provider)
    generator = DeterministicGenerator(data_provider, processor)
    dataset = generator.generate_dataset()
    # train knn classifier
    classifier = KNNClassifier(data_provider, dataset, DEVICE)
    classifier.n_neighbours = 5
    classifier.train()
    # test the classifier
    test = ClassificationTest(dataset, classifier)
    print(str(num_features) + " Features: " + str(test.test()) + "% Präzision")