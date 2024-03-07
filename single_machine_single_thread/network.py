import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(torch.nn.Module):
    def __init__(self, state_size, action_size,hidden_size):
        super(Policy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2*hidden_size)
        self.linear3 = nn.Linear(2*hidden_size, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = torch.distributions.Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(torch.nn.Module):
    def __init__(self, state_size, action_size,hidden_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2*hidden_size)
        self.linear3 = nn.Linear(2*hidden_size, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value