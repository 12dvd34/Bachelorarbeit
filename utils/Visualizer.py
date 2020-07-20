import matplotlib.pyplot as plt
# generates a chart to show progress of training
# not legend or axis descriptions though


class Visualizer:
    def __init__(self):
        self.fig, self.ax_loss = plt.subplots()
        self.ax_prec = self.ax_loss.twinx()
        self.epochs_loss = []
        self.losses = []
        self.epochs_prec = []
        self.prec = []
        self.epochs_train_prec = []
        self.train_prec = []

    def add_loss(self, epoch, loss):
        self.epochs_loss.append(epoch)
        self.losses.append(loss)

    def add_prec(self, epoch, prec):
        self.epochs_prec.append(epoch)
        self.prec.append(prec)

    def add_train_prec(self, epoch, prec):
        self.epochs_train_prec.append(epoch)
        self.train_prec.append(prec)

    def show(self):
        self.ax_loss.plot(self.epochs_loss, self.losses)
        if len(self.prec) > 0:
            self.ax_prec.plot(self.epochs_prec, self.prec, "g-")
        if len(self.train_prec) > 0:
            self.ax_prec.plot(self.epochs_train_prec, self.train_prec, "r-")
        self.fig.show()