import torch
# can be used to save the neural network for later reuse


def save(classifier, path):
    data = {
        "epoch": classifier.epoch,
        "state": classifier.model.state_dict(),
        "optimizer": classifier.optimizer.state_dict()
    }
    torch.save(data, path)


def load(classifier, path):
    data = torch.load(path)
    classifier.model.load_state_dict(data["state"])
    classifier.optimizer.load_state_dict(data["optimizer"])
    classifier.epoch = data["epoch"]