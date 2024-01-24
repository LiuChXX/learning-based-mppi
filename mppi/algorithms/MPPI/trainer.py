import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


class Trainer:
    def __init__(self, net, epochs, batch_size, lr_init, lr_decay_rate, replaybuffer, device=torch.device("cpu")):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_init = lr_init
        self.lr_decay_rate = lr_decay_rate
        self.net = net
        self.device = device
        self.ReplayBuffer = replaybuffer

    def train(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr_init)
        lr_scheduder = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay_rate)
        loss_fn = nn.MSELoss()

        for epoch in range(self.epochs):
            train_data = self.ReplayBuffer.buffer
            n_batch = len(train_data)//self.batch_size
            train_loss_sum = 0

            for i in range(n_batch):
                transitions = self.ReplayBuffer.buffer[i * self.batch_size: (i+1) * self.batch_size]
                # transitions = self.ReplayBuffer.sample(self.batch_size)
                batch_state, batch_actions, batch_reward, batch_next_state, batch_done = zip(*transitions)

                # list to np.ndarray
                batch_state = np.array(batch_state)
                batch_actions = np.array(batch_actions)
                batch_next_state = np.array(batch_next_state)

                predict_next_obs = self.net.dynamic(batch_state, batch_actions)
                batch_next_state = torch.tensor(batch_next_state, device=self.device, dtype=torch.float32)

                loss = loss_fn(predict_next_obs, batch_next_state)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()

            lr_scheduder.step()
            print('[%d,%d] loss:%.07f' % (epoch + 1, self.epochs, train_loss_sum / n_batch))
