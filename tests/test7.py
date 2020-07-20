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
HIDDEN_SIZE = 10000
BATCH_SIZE = [16, 32, 64, 128, 256]
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
for batch_size in BATCH_SIZE:
    print("batch size: " + str(batch_size))
    # creating classifier, overwriting parameters
    classifier = FFNClassifier(data_provider, dataset, DEVICE, batch_size=batch_size, hidden_size=HIDDEN_SIZE)
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