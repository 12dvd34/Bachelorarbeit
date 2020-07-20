# tests the given classifier on the given dataset

class ClassificationTest:
    def __init__(self, dataset, classifier):
        self.dataset = dataset
        self.classifier = classifier

    def test(self):
        correct = 0
        for input, output in self.dataset.testset:
            correct += self.classifier.classify(input) == output.item()
        return round(correct / len(self.dataset.testset) * 100, 2)

    def test_trainset(self):
        correct = 0
        for input, output in self.dataset.trainset:
            correct += self.classifier.classify(input) == output.item()
        return round(correct / len(self.dataset.trainset) * 100, 2)
