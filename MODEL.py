import torch
import torch.nn as nn


class duel_dqn(nn.Module):
    def __init__(self, n_frame, n_action, device):
        super(duel_dqn, self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 16, 8, 4)
        self.layer2 = nn.Conv2d(16, 16, 3, 1)
        self.fc = nn.Linear(5184, 128)
        self.q = nn.Linear(128, n_action)
        self.v = nn.Linear(128, 1)

        self.device = device
        self.seq = nn.Sequential(self.layer1, self.layer2, self.fc, self.q, self.v)

        self.seq.apply(init_weights)

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x).to(self.device)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = x.view(-1, 5184)
        x = torch.relu(self.fc(x))
        adv = self.q(x)
        v = self.v(x)
        q = v + (adv - 1 / adv.shape[-1] * adv.max(-1, True)[0])

        return q


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)