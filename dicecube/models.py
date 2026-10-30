"""
Value + policy network for the dice cube.

Same residual-MLP backbone as PylosNetwork. The heads differ because this is
a single-player puzzle rather than a two-player game:

  value  -- estimated cost-to-go (moves left to ANY solved state), >= 0
  policy -- distribution over the 12 quarter turns

The network takes raw sticker values (B, 54) and one-hot encodes them itself
(see dicecube.cube.to_observation for the numpy reference encoding) -- doing
it in torch is ~6x faster than numpy, and training evaluates ~12k states per
step.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from dicecube.cube import NON_CENTER, OBS_DIM, PIECE_VALUES
from engine.models import ResidualBlock

_VALUE_INDEX = torch.zeros(10, dtype=torch.long)
_VALUE_INDEX[list(PIECE_VALUES)] = torch.arange(8)
_NON_CENTER = torch.as_tensor(NON_CENTER)


class DiceCubeNetwork(nn.Module):
    def __init__(self, input_dim=OBS_DIM, action_space=12, hidden=512, num_blocks=4,
                 value_hidden=128, policy_hidden=128, device=None):
        super().__init__()
        self.config = {
            "input_dim": input_dim, "action_space": action_space, "hidden": hidden,
            "num_blocks": num_blocks, "value_hidden": value_hidden,
            "policy_hidden": policy_hidden,
        }

        self.input_fc = nn.Linear(input_dim, hidden)
        self.input_bn = nn.BatchNorm1d(hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(num_blocks)])

        self.value_fc1 = nn.Linear(hidden, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

        self.policy_fc1 = nn.Linear(hidden, policy_hidden)
        self.policy_fc2 = nn.Linear(policy_hidden, action_space)

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = torch.device(device)
        self.to(self.device)

    def encode(self, states):
        """(B, 54) sticker values -> (B, OBS_DIM) float features."""
        idx = _VALUE_INDEX.to(states.device)[states.long()[:, _NON_CENTER.to(states.device)]]
        one_hot = F.one_hot(idx, 8).float()                 # (B, 48, 8)
        hist = one_hot.view(-1, 6, 8, 8).sum(dim=2)         # (B, 6, 8)
        return torch.cat([one_hot.flatten(1), hist.flatten(1)], dim=1)

    def _backbone(self, x):
        x = F.relu(self.input_bn(self.input_fc(x)))
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, states):
        x = self._backbone(self.encode(states))
        cost = F.softplus(self.value_fc2(F.relu(self.value_fc1(x)))).squeeze(-1)
        log_policy = F.log_softmax(self.policy_fc2(F.relu(self.policy_fc1(x))), dim=-1)
        return cost, log_policy

    def batched_inference(self, states_np):
        """numpy (B, 54) states -> (cost (B,), policy (B, 12)) as numpy."""
        self.eval()
        with torch.no_grad():
            states = torch.as_tensor(states_np, device=self.device)
            cost, log_policy = self(states)
            return cost.cpu().numpy(), log_policy.exp().cpu().numpy()


def save_model(model, path, extra=None):
    # Write-then-rename so a server reloading the file never sees half of it
    tmp = path + ".tmp"
    torch.save({"model_state_dict": model.state_dict(), "model_config": model.config,
                **(extra or {})}, tmp)
    os.replace(tmp, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = DiceCubeNetwork(device=device, **ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt
