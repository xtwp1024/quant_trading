"""
Deep Q-Network architectures for share-sizing trading agents.

Adapted from deep-q-trading-agent (Jeong et al., 2019)
https://www.sciencedirect.com/science/article/abs/pii/S0957417418306134

Three architectures:
- NumQ:        Joint Q-network (single branch for action + share sizing)
- NumDReg-AD:  Action-Dependent DNN Regressor (share sizing depends on action)
- NumDReg-ID:  Action-Independent DNN Regressor (share sizing independent of action)

Key innovation: Networks output BOTH action (buy/sell/hold) AND share quantity,
rather than just the action -- making it a more realistic trading agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

# Method constants
NUMQ = 0
NUMDREG_AD = 1
NUMDREG_ID = 2

# Mode constants for NumDReg training
ACT_MODE = 0   # Optimize action branch only
NUM_MODE = 1   # Optimize number branch only
FULL_MODE = 2  # Optimize both branches

# Action constants
BUY = 0
HOLD = 1
SELL = 2

torch.set_default_dtype(torch.float64)


class NumQModel(nn.Module):
    """
    Joint Q-network: Single branch outputs both Q-values and share ratios.

    Architecture:
        fc1(200 -> 200) -> relu
        fc2(200 -> 100) -> relu
        fc3(100 -> 50)  -> relu
        fc_q(50 -> 3)   -> Q values for BUY, HOLD, SELL
        softmax(fc_q(sigmoid(fc3))) -> share ratios

    Returns:
        q:     Q-values for each action  (shape: [3])
        r:     Share ratios (softmax)   (shape: [3])
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=200, out_features=200, bias=True)
        self.fc2 = nn.Linear(in_features=200, out_features=100, bias=True)
        self.fc3 = nn.Linear(in_features=100, out_features=50, bias=True)
        self.fc_q = nn.Linear(in_features=50, out_features=3, bias=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        q = self.fc_q(F.relu(x))
        r = F.softmax(self.fc_q(torch.sigmoid(x)), dim=-1)

        return q, r


class NumDRegModel(nn.Module):
    """
    Dual-branch DNN Regressor for action and share sizing.

    NumDReg-AD: share ratio depends on selected action (3 outputs)
    NumDReg-ID: share ratio independent of action (1 output)

    Architecture:
        Root:     fc1(200 -> 100) -> relu

        Action branch:
        fc2_act(100 -> 50) -> relu
        fc3_act(50 -> 20)  -> relu
        fc_q(20 -> 3)      -> Q values

        Number branch:
        fc2_num(100 -> 50) -> relu
        fc3_num(50 -> 20)  -> sigmoid
        fc_r(20 -> 3)      -> share ratios (AD, softmax)
        fc_r(20 -> 1)      -> share ratio (ID, sigmoid)

    Returns:
        q: Q-values for each action  (shape: [3])
        r: Share ratios              (shape: [3] for AD, [1] for ID)
    """

    def __init__(self, method: int, mode: int = FULL_MODE):
        super().__init__()

        self.method = method  # NUMDREG_AD or NUMDREG_ID
        self.mode = mode      # ACT_MODE, NUM_MODE, or FULL_MODE

        # Root shared layer
        self.fc1 = nn.Linear(in_features=200, out_features=100, bias=True)

        # Action branch
        self.fc2_act = nn.Linear(in_features=100, out_features=50, bias=True)
        self.fc3_act = nn.Linear(in_features=50, out_features=20, bias=True)
        self.fc_q = nn.Linear(in_features=20, out_features=3, bias=True)

        # Number branch
        self.fc2_num = nn.Linear(in_features=100, out_features=50, bias=True)
        self.fc3_num = nn.Linear(in_features=50, out_features=20, bias=True)
        # AD: 3 outputs (per action), ID: 1 output (independent)
        num_outputs = 3 if self.method == NUMDREG_AD else 1
        self.fc_r = nn.Linear(in_features=20, out_features=num_outputs, bias=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Root
        x = F.relu(self.fc1(x))

        # Action branch
        x_act = F.relu(self.fc2_act(x))
        x_act = F.relu(self.fc3_act(x_act))
        q = self.fc_q(x_act)

        # Number branch
        if self.mode == ACT_MODE:
            # During action training step: derive shares from action branch
            r = F.softmax(self.fc_q(torch.sigmoid(x_act)), dim=-1)
        else:
            x_num = F.relu(self.fc2_num(x))
            x_num = torch.sigmoid(self.fc3_num(x_num))
            if self.method == NUMDREG_ID:
                r = torch.sigmoid(self.fc_r(x_num))
            else:
                r = F.softmax(self.fc_r(x_num), dim=-1)

        return q, r

    def set_mode(self, mode: int):
        """Set training mode: ACT_MODE, NUM_MODE, or FULL_MODE."""
        self.mode = mode


class StonksNet(nn.Module):
    """
    Autoencoder for stock price prediction / grouping.

    Used for transfer learning: trains on component stocks to learn
    compressed representations, then used to group stocks by prediction error.

    Architecture:
        fc1(size -> 5) -> relu
        out(5 -> size)  -> relu
    """

    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.fc1 = nn.Linear(in_features=size, out_features=5, bias=True)
        self.out = nn.Linear(in_features=5, out_features=size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.out(x))
        return x
