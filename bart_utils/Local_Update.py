import torch
from torch import nn

class client_train(object):
    def __init__(self, args, device, lr, weight_decay, dataloader):
        self.args = args
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = dataloader

    def train(self, net):
        net = net.to(self.device)
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        epoch_loss = []

        for epoch in range(self.args.n_local_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
