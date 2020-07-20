import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import utils.Batching as Batching
# word2vec network using pseudoepochs

# depends on gpu memory and feature size
# influence on result not tested
num_tokens_at_once = 20


class Net(nn.Module):
    def __init__(self, vocab_size, num_features=300):
        super(Net, self).__init__()
        self.hidden = nn.Linear(vocab_size, num_features)
        self.output = nn.Linear(num_features, vocab_size)

    def forward(self, x):
        x = self.hidden(x)
        x = F.softmax(self.output(x), 0)
        return x


class Word2Vec:
    def __init__(self, data_provider, num_features=300, device_name="cpu"):
        self.data_provider = data_provider
        self.device = torch.device(device_name)
        self.net = Net(len(data_provider.token_mapping), num_features).to(self.device)
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters())

    def train(self, epochs=10, batch_size=1):
        all_tokens = list(self.data_provider.token_mapping.keys())
        for i in range(int((len(all_tokens) / num_tokens_at_once) + 1)):
            tokens_to_train = all_tokens[(i*num_tokens_at_once):(i*num_tokens_at_once+num_tokens_at_once)]
            inputs = []
            targets = []
            for token in tokens_to_train:
                for shared in self.data_provider.find_shared_tokens(token):
                    inputs.append(self.data_provider.generate_token_one_hot(token))
                    targets.append(self.data_provider.generate_token_one_hot(shared))
            input_batches = Batching.generate_batches(inputs, batch_size)
            target_batches = Batching.generate_batches(targets, batch_size)
            for epoch in range(epochs):
                epoch_loss = 0
                count = 0
                for i in range(len(input_batches)):
                    inpt = input_batches[i].to(self.device)
                    targt = target_batches[i].to(self.device)
                    self.optimizer.zero_grad()
                    output = self.net(inpt)
                    loss = self.criterion(output, targt)
                    loss.backward()
                    epoch_loss += loss.item()
                    count += 1
                    self.optimizer.step()
                epoch_loss /= count
            del inputs
            del targets
            del input_batches
            del target_batches
            del tokens_to_train

    def save_weights(self, file):
        weights = list(self.net.hidden.parameters())[0].tolist()
        file.write(json.dumps(weights))
        file.close()
        return weights

    def get_weights(self):
        return list(self.net.hidden.parameters())[0].tolist()
