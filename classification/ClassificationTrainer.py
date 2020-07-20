from utils.Visualizer import Visualizer
from classification.ClassificationTest import ClassificationTest
# provides information on progress while training a classifier


class ClassificationTrainer:
    def __init__(self, classifier):
        self.classifier = classifier
        self.visualizer = Visualizer()
        self.test = ClassificationTest(self.classifier.dataset, self.classifier)
        self.loss_interval = 0
        self.print_loss = False
        self.precision_interval = 0
        self.print_precision = False
        self.trainset_precision_interval = 0
        self.print_trainset_precision = False

    # decide how often loss should be logged, and if it should be printed to console
    def enable_loss(self, interval=1, print=False):
        self.loss_interval = interval
        self.print_loss = print

    # decide how often precision should be logged, and if it should be printed to console
    def enable_precision(self, interval=5, print=False):
        self.precision_interval = interval
        self.print_precision = print

    # decide how often loss shall be should, and if it should be printed to console
    def enable_trainset_precision(self, interval=10, print=False):
        self.trainset_precision_interval = interval
        self.print_trainset_precision = print

    # trains the classifier and prints info as configured
    def train(self, epochs=10):
        for _ in range(epochs):
            loss = self.classifier.train_epoch()
            epoch = self.classifier.epoch
            if self.loss_interval != 0:
                if epoch % self.loss_interval == 0:
                    self.visualizer.add_loss(epoch, loss)
                    if self.print_loss:
                        print("[" + str(epoch) + "] " + "Loss: " + str(loss))
            if self.precision_interval != 0:
                if epoch % self.precision_interval == 0:
                    precision = self.test.test()
                    self.visualizer.add_prec(epoch, precision)
                    if self.print_precision:
                        print("[" + str(epoch) + "] " + "Precision: " + str(precision))
            if self.trainset_precision_interval != 0:
                if epoch % self.trainset_precision_interval == 0:
                    trainset_precision = self.test.test_trainset()
                    self.visualizer.add_train_prec(epoch, trainset_precision)
                    if self.print_trainset_precision:
                        print("[" + str(epoch) + "] " + "Trainset Precision: " + str(trainset_precision))

    # shows a diagram of the training progress
    def show(self):
        self.visualizer.show()
