from utils.StopWatch import StopWatch
from utils.RawDataProcessing import TokenVectorProcessor
from utils.DataProviderLight import DataProviderLight
from utils.DatasetGeneration import DeterministicGenerator
from classification.ClassificationTest import ClassificationTest
from classification.RNNClassification import RNNClassifier
from classification.ClassificationTrainer import ClassificationTrainer
# script for corresponding test case
# most test cases should be able to be executed without any further changes, if data is available

SAMPLE_SIZE = 1000
HIDDEN_SIZE = [100, 200, 500, 1000, 2000]
BATCH_SIZE = 32
EPOCHS = 100
# change device to "cpu" if cuda not available
DEVICE = "cuda"
stopwatch = StopWatch()
# pregenerated embedding and labels
file_words = open("../data/unique_equations.json")
file_labels = open("../data/unique_labels.json")
file_weights = open("../tests/matrix1.json")
data_provider = DataProviderLight(file_words, file_labels, sample_size=SAMPLE_SIZE, file_weights=file_weights)
# embedding data, splitting up into train and test set
processor = TokenVectorProcessor(data_provider, padding=False)
generator = DeterministicGenerator(data_provider, processor)
dataset = generator.generate_dataset()
for hidden_size in HIDDEN_SIZE:
    print("State Size: " + str(hidden_size))
    # creating classifier
    classifier = RNNClassifier(data_provider, dataset, DEVICE, batch_size=BATCH_SIZE, hidden_size=hidden_size)
    # train classifier and output progress
    trainer = ClassificationTrainer(classifier)
    trainer.enable_loss(10, True)
    stopwatch.start()
    trainer.train(EPOCHS)
    stopwatch.stop()
    # testing
    test = ClassificationTest(dataset, classifier)
    print("Präzision: " + str(test.test()) + "%")
    print("Präzision (Trainingsset): " + str(test.test_trainset()) + "%")