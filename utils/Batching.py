import torch
# wraps the data to batches
# doesn't work for RNNs


def generate_batches(data, batch_size=10):
    dtype = data[0].dtype
    batched_data = []
    for i in range(len(data))[::batch_size]:
        l = torch.zeros(len(data[i:i + batch_size]), len(data[0]), dtype=dtype)
        for j, elem in enumerate(data[i:i + batch_size]):
            l[j] = elem
        batched_data.append(l)
    return batched_data


class BatchedDataset:
    def __init__(self, dataset, batch_size=10):
        self.trainset_inputs = []
        self.trainset_outputs = []
        for tup in dataset.trainset:
            self.trainset_inputs.append(tup[0])
            self.trainset_outputs.append(tup[1])
        self.trainset_inputs = generate_batches(self.trainset_inputs, batch_size)
        self.trainset_outputs = generate_batches(self.trainset_outputs, batch_size)
        self.testset_inputs = []
        self.testset_outputs = []
        for tup in dataset.testset:
            self.testset_inputs.append(tup[0])
            self.testset_outputs.append(tup[1])
