import torch

from utils.DatasetGeneration import DeterministicGenerator
from utils.RawDataProcessing import VectorProcessor
from utils.StopWatch import StopWatch
from utils.DataProviderLight import DataProviderLight
from preprocessing.Word2VecEpochs import Token2Vec
from preprocessing.Word2Vec import Word2Vec
from classification.KNNClassification import KNNClassifier
from classification.ClassificationTest import ClassificationTest
# script for corresponding test case
# most test cases should be able to be executed without any further changes, if data is available

FEATURES = 100
SAMPLE_SIZE = 10000
EPOCHS = 5
BATCH_SIZE = 32
# change device to "cpu" if cuda not available
DEVICE = "cuda"
stopwatch = StopWatch()
# pregenerated embedding and labels
file_words = open("../data/unique_equations.json")
file_labels = open("../data/unique_labels.json")
file_weights = open("../data/weights_0.json")
data_provider = DataProviderLight(file_words, file_labels, sample_size=SAMPLE_SIZE, file_weights=file_weights)
processor = VectorProcessor(data_provider)
generator = DeterministicGenerator(data_provider, processor)

w2v_epochs = Word2Vec(data_provider, FEATURES, DEVICE)
stopwatch.start()
w2v_epochs.train(EPOCHS, BATCH_SIZE)
stopwatch.stop()
data_provider.weights = torch.tensor(w2v_epochs.get_weights())
dataset = generator.generate_dataset()
classifier = KNNClassifier(data_provider, dataset, DEVICE)
classifier.n_neighbours = 5
classifier.train()
test = ClassificationTest(dataset, classifier)
print("Präzision: " + str(test.test()) + "%")