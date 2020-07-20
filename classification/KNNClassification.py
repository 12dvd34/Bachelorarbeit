import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# wraps the KNN classifier for uniform access


class KNNClassifier:
    def __init__(self, data_provider, dataset, device_name="cpu"):
        # data_provider, device_name and epoch are only for compatibility, not used
        self.epoch = 0
        # n_neighbours can be overwritten externally, but 1 should be fine
        self.n_neighbours = 1
        self.data_list = []
        self.label_list = []
        for data, label in dataset.trainset:
            self.data_list.append(data.tolist())
            self.label_list.append(label.item())
        # shouldn't do anything, too afraid it will break something when removed
        self.knn_classifier = KNeighborsClassifier(n_neighbors=self.n_neighbours)

    def train_epoch(self):
        # not an actual epoch, but has to be compatible with other classifiers
        self.knn_classifier = KNeighborsClassifier(n_neighbors=self.n_neighbours)
        self.knn_classifier.fit(self.data_list, self.label_list)
        # return of loss is expected, but there is none
        return 404

    def train(self, epochs=1):
        # just for compatibility
        self.train_epoch()

    def classify(self, input):
        return self.knn_classifier.predict([np.asarray(input)])[0]
