import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import Batching
import json
# word2vec network using real epochs
# can also simulate pseudoepochs, not tested though

# amount of in- and output tuples held in storage at the same time
# 500.000 on 8 GB with 100 Features
TUPLE_LIMIT = 500000


class Net(nn.Module):
    def __init__(self, alphabet_size, num_features):
        super(Net, self).__init__()
        self.hidden = nn.Linear(alphabet_size, num_features)
        self.out = nn.Linear(num_features, alphabet_size)

    def forward(self, x):
        x = self.hidden(x)
        x = F.softmax(self.out(x), 0)
        return x


class Token2Vec:
    def __init__(self, data_provider, num_features=50, device_name="cpu"):
        self.data_provider = data_provider
        self.device = torch.device(device_name)
        alphabet_size = len(self.data_provider.token_mapping)
        self.model = Net(alphabet_size, num_features).to(self.device)
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.inputs = []
        self.targets = []

    def train(self, epochs=10, batch_size=10, pseudo_epochs=1):
        self.inputs = []
        self.targets = []
        for epoch in range(epochs):
            for token in self.data_provider.token_mapping.keys():
                for shared in self.data_provider.find_shared_tokens(token):
                    self.inputs.append(self.data_provider.generate_token_one_hot(token))
                    self.targets.append(self.data_provider.generate_token_one_hot(shared))
                    if len(self.inputs) == TUPLE_LIMIT:
                        self.inputs = Batching.generate_batches(self.inputs, batch_size)
                        self.targets = Batching.generate_batches(self.targets, batch_size)
                        self.train_tuples(pseudo_epochs)
        self.train_tuples(pseudo_epochs)

    def train_tuples(self, pseudo_epochs):
        for _ in range(pseudo_epochs):
            for i in range(len(self.inputs)):
                input = self.inputs[i].to(self.device)
                target = self.targets[i].to(self.device)
                self.optimizer.zero_grad()
                output = self.model(input)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        for _ in range(len(self.inputs)):
            del self.inputs[0]
            del self.targets[0]

    def save_weights(self, file):
        weights = list(self.model.hidden.parameters())[0].tolist()
        file.write(json.dumps(weights))
        file.close()
        return weights

    def get_weights(self):
        return list(self.model.hidden.parameters())[0].tolist()
