import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Pre-activation residual block: BN -> ReLU -> Linear -> BN -> ReLU -> Linear + skip."""

    def __init__(self, dim):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(x))
        out = self.fc1(out)
        out = F.relu(self.bn2(out))
        out = self.fc2(out)
        return out + residual


class PylosNetwork(nn.Module):
    def __init__(self, input_shape, action_space, hidden=256, num_blocks=6):
        super().__init__()

        # Input projection
        self.input_fc = nn.Linear(input_shape[0], hidden)
        self.input_bn = nn.BatchNorm1d(hidden)

        # Residual tower
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(num_blocks)])

        # Value head
        self.value_fc1 = nn.Linear(hidden, 64)
        self.value_bn = nn.BatchNorm1d(64)
        self.value_fc2 = nn.Linear(64, 1)

        # Policy head
        self.policy_fc1 = nn.Linear(hidden, 128)
        self.policy_bn = nn.BatchNorm1d(128)
        self.policy_fc2 = nn.Linear(128, action_space)

        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

    def _backbone(self, x):
        """Shared residual backbone."""
        x = F.relu(self.input_bn(self.input_fc(x)))
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, observations):
        x = self._backbone(observations)
        # Value head
        v = F.relu(self.value_bn(self.value_fc1(x)))
        value = torch.tanh(self.value_fc2(v))
        # Policy head
        p = F.relu(self.policy_bn(self.policy_fc1(x)))
        log_policy = F.log_softmax(self.policy_fc2(p), dim=-1)
        return value, log_policy

    def inference(self, observation):
        """Combined value + policy in a single forward pass."""
        self.eval()
        with torch.no_grad():
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)
            x = self._backbone(observation)
            v = F.relu(self.value_bn(self.value_fc1(x)))
            value = torch.tanh(self.value_fc2(v))
            p = F.relu(self.policy_bn(self.policy_fc1(x)))
            policy = F.softmax(self.policy_fc2(p), dim=-1)
            return value.squeeze(0).item(), policy.squeeze(0).cpu().numpy()

    def batched_inference(self, observations_np):
        """Batch inference for multiple observations."""
        self.eval()
        with torch.no_grad():
            obs = torch.tensor(observations_np, device=self.device)
            x = self._backbone(obs)
            v = F.relu(self.value_bn(self.value_fc1(x)))
            values = torch.tanh(self.value_fc2(v)).squeeze(-1)
            p = F.relu(self.policy_bn(self.policy_fc1(x)))
            policies = F.softmax(self.policy_fc2(p), dim=-1)
            return values.cpu().numpy(), policies.cpu().numpy()

    def value_forward(self, observation):
        self.eval()
        with torch.no_grad():
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)
            x = self._backbone(observation)
            v = F.relu(self.value_bn(self.value_fc1(x)))
            return torch.tanh(self.value_fc2(v)).squeeze(0)

    def policy_forward(self, observation):
        self.eval()
        with torch.no_grad():
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)
            x = self._backbone(observation)
            p = F.relu(self.policy_bn(self.policy_fc1(x)))
            return F.softmax(self.policy_fc2(p), dim=-1).squeeze(0)
