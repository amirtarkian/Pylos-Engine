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

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, observations):
        x = F.relu(self.fc1(observations))
        x = F.relu(self.fc2(x))
        value = torch.tanh(self.value_head(x))
        log_policy = F.log_softmax(self.policy_head(x), dim=-1)
        return value, log_policy

    def value_forward(self, observation):
        with torch.no_grad():
            x = F.relu(self.fc1(observation))
            x = F.relu(self.fc2(x))
            return torch.tanh(self.value_head(x))

    def policy_forward(self, observation):
        with torch.no_grad():
            x = F.relu(self.fc1(observation))
            x = F.relu(self.fc2(x))
            return F.softmax(self.policy_head(x), dim=-1)

    def inference(self, observation):
        """Combined value + policy in a single forward pass (no mode toggling)."""
        with torch.no_grad():
            x = F.relu(self.fc1(observation))
            x = F.relu(self.fc2(x))
            value = torch.tanh(self.value_head(x))
            policy = F.softmax(self.policy_head(x), dim=-1)
            return value.item(), policy.cpu().numpy()

    def batched_inference(self, observations_np):
        """Batch inference for multiple observations in a single forward pass.

        Args:
            observations_np: numpy array of shape (N, input_dim)

        Returns:
            (values, policies): numpy arrays of shape (N,) and (N, action_space)
        """
        self.eval()
        with torch.no_grad():
            obs = torch.tensor(observations_np, device=self.device)
            x = F.relu(self.fc1(obs))
            x = F.relu(self.fc2(x))
            values = torch.tanh(self.value_head(x)).squeeze(-1)
            policies = F.softmax(self.policy_head(x), dim=-1)
            return values.cpu().numpy(), policies.cpu().numpy()
