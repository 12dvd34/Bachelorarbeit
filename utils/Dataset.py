# convenient wrapper for training- and testset to be passed around together


class Dataset:
    def __init__(self):
        self.trainset = []
        self.testset = []

    def get_trainset_input(self, index):
        return self.trainset[index][0]

    def get_trainset_output(self, index):
        return self.trainset[index][1]

    def get_testset_input(self, index):
        return self.testset[index][0]

    def get_testset_output(self, index):
        return self.testset[index][1]
