import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# wraps the RNN classifier for uniform access


class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Net, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.rnn(x)
        x = x[0][x[0].size(0) - 1]
        x = self.out(x)
        x = F.softmax(x, 1)
        return x


class RNNClassifier:
    def __init__(self, data_provider, dataset, device_name="cpu", batch_size=32, hidden_size=200):
        self.data_provider = data_provider
        self.dataset = dataset
        self.device = torch.device(device_name)
        self.epoch = 0
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        # get data dependent parameters
        input_size = dataset.get_trainset_input(0).size(1)
        output_size = len(data_provider.label_mapping)
        # create net, criterion and optimizer
        self.model = Net(input_size, output_size, hidden_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), 0.1)

    def train(self, epochs=10):
        for epoch in range(epochs):
            self.train_epoch()

    def train_epoch(self):
        epoch_loss = 0
        for i in range(len(self.dataset.trainset))[::self.batch_size]:
            self.optimizer.zero_grad()
            batch_inputs = self.dataset.get_trainset_input(i).to(self.device)
            batch_inputs = batch_inputs.view(batch_inputs.size(0), 1, batch_inputs.size(1))
            batch_targets = self.dataset.get_trainset_output(i).to(self.device)
            for j in range(1, self.batch_size):
                index = i + j
                if index >= len(self.dataset.trainset):
                    break
                input = self.dataset.get_trainset_input(index)
                input = input.view(input.size(0), 1, input.size(1)).to(self.device)
                target = self.dataset.get_trainset_output(index).to(self.device)
                batch_inputs = torch.cat((batch_inputs, input), 1)
                batch_targets = torch.cat((batch_targets, target), 0)
            output = self.model(batch_inputs)
            loss = self.criterion(output, batch_targets)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        self.epoch += 1
        return epoch_loss / len(self.dataset.trainset)

    def classify(self, input):
        with torch.no_grad():
            input = input.view(input.size(0), 1, input.size(1)).to(self.device)
            print(torch.argmax(self.model(input.to(self.device))).item())
            return torch.argmax(self.model(input.to(self.device))).item()