import torch
import torch.nn as nn
import torch.nn.functional as F


class PylosNetwork(nn.Module):
    def __init__(self, input_shape, action_space, hidden1=512, hidden2=256):
        super().__init__()
        self.fc1 = nn.Linear(input_shape[0], hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.value_head = nn.Linear(hidden2, 1)
        self.policy_head = nn.Linear(hidden2, action_space)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def __call__(self, observations):
        self.train()
        x = F.relu(self.fc1(observations))
        x = F.relu(self.fc2(x))
        value = torch.tanh(self.value_head(x))
        log_policy = F.log_softmax(self.policy_head(x), dim=-1)
        return value, log_policy

    def value_forward(self, observation):
        self.eval()
        with torch.no_grad():
            x = F.relu(self.fc1(observation))
            x = F.relu(self.fc2(x))
            return torch.tanh(self.value_head(x))

    def policy_forward(self, observation):
        self.eval()
        with torch.no_grad():
            x = F.relu(self.fc1(observation))
            x = F.relu(self.fc2(x))
            return F.softmax(self.policy_head(x), dim=-1)
