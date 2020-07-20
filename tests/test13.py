from utils.StopWatch import StopWatch
from utils.RawDataProcessing import PregeneratedProcessor
from utils.DataProviderLight import DataProviderLight
from utils.DatasetGeneration import DeterministicGenerator
from classification.FFNClassification import FFNClassifier
from classification.ClassificationTrainer import ClassificationTrainer
# script for corresponding test case
# most test cases should be able to be executed without any further changes, if data is available

SAMPLE_SIZE = 100000
EPOCHS = 150
HIDDEN_SIZE = 10000
BATCH_SIZE = 128
# change device to "cpu" if cuda not available
DEVICE = "cuda"
stopwatch = StopWatch()
# pregenerated embedding and labels
file_words = open("../tests/embedding8.json")
file_labels = open("../data/unique_labels.json")
data_provider = DataProviderLight(file_words, file_labels, sample_size=SAMPLE_SIZE)
# embedding data, splitting up into train and test set
processor = PregeneratedProcessor(data_provider)
generator = DeterministicGenerator(data_provider, processor)
dataset = generator.generate_dataset()
# creating classifier, overwriting parameters
classifier = FFNClassifier(data_provider, dataset, DEVICE, BATCH_SIZE, HIDDEN_SIZE)
# train classifier and output progress
trainer = ClassificationTrainer(classifier)
trainer.enable_loss(10, True)
trainer.enable_precision(10, True)
trainer.enable_trainset_precision(10, True)
trainer.train(EPOCHS)
trainer.show()
print("done")