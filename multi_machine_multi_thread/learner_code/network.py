import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self,state_size,action_size,hidden_size):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_size)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.orthogonal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        output = F.relu(self.linear1(x))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = torch.distributions.Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self,state_size,output_size,hidden_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.orthogonal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        output = F.relu(self.linear1(x))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value