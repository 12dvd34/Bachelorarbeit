from utils.StopWatch import StopWatch
from utils.RawDataProcessing import PregeneratedProcessor
from utils.DataProviderLight import DataProviderLight
from utils.DatasetGeneration import DeterministicGenerator
from classification.ClassificationTest import ClassificationTest
from classification.FFNClassification import FFNClassifier
from classification.ClassificationTrainer import ClassificationTrainer
# script for corresponding test case
# most test cases should be able to be executed without any further changes, if data is available

SAMPLE_SIZE = 10000
HIDDEN_SIZE = [1000, 2000, 5000, 10000, 20000]
BATCH_SIZE = 64
EPOCHS = 100
# change device to "cpu" if cuda not available
DEVICE = "cuda"
stopwatch = StopWatch()
# pregenerated embedding and labels
file_words = open("../tests/embedding1.json")
file_labels = open("../data/unique_labels.json")
data_provider = DataProviderLight(file_words, file_labels, sample_size=SAMPLE_SIZE)
# embedding data, splitting up into train and test set
processor = PregeneratedProcessor(data_provider)
generator = DeterministicGenerator(data_provider, processor)
dataset = generator.generate_dataset()
for hidden_size in HIDDEN_SIZE:
    print("hidden size: " + str(hidden_size))
    # creating classifier, overwriting parameters
    classifier = FFNClassifier(data_provider, dataset, DEVICE, batch_size=BATCH_SIZE, hidden_size=hidden_size)
    # train classifier and output progress
    trainer = ClassificationTrainer(classifier)
    trainer.enable_loss(10, True)
    stopwatch.start()
    trainer.train(EPOCHS)
    stopwatch.stop()
    # test not really necessary, done by trainer
    test = ClassificationTest(dataset, classifier)
    print("Präzision: " + str(test.test()) + "%")
    print("Präzision (Trainingsset): " + str(test.test_trainset()) + "%")