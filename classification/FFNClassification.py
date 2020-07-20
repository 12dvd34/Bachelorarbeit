import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.Batching import BatchedDataset
# wraps the FFN classifier for uniform access


class FFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        output = self.out(x)
        return output


class FFNClassifier:
    def __init__(self, data_provider, dataset, device_name="cpu", batch_size=32, hidden_size=10000):
        self.dataset = dataset
        self.device = torch.device(device_name)
        self.epoch = 0
        # get model parameters
        input_size = len(dataset.get_trainset_input(0))
        output_size = len(data_provider.label_mapping)
        # create model, loss and optimizer
        self.model = FFN(input_size, hidden_size, output_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        # get inputs and targets from dataset
        batched_dataset = BatchedDataset(dataset, batch_size)
        self.inputs = batched_dataset.trainset_inputs
        self.targets = batched_dataset.trainset_outputs

    def train(self, epochs=10):
        for epoch in range(epochs):
            self.train_epoch()

    def train_epoch(self):
        epoch_loss = 0
        for index in range(len(self.inputs)):
            self.optimizer.zero_grad()
            input = self.inputs[index].to(self.device)
            target = self.targets[index].to(self.device)
            output = self.model(input)
            loss = self.criterion(output, target.view(target.size(0)))
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        self.epoch += 1
        return epoch_loss / len(self.inputs)

    def classify(self, input):
        with torch.no_grad():
            return torch.argmax(self.model(input.to(self.device))).item()
