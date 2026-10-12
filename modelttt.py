import torch
import torch.nn as nn
import torch.nn.functional as F

class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        # Input: 3 channels x 3x3 grid
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Policy Head (Outputs 9 probabilities)
        self.policy_fc = nn.Linear(64 * 3 * 3, 9) 
        
        # Value Head (Outputs 1 win/loss prediction)
        self.value_fc1 = nn.Linear(64 * 3 * 3, 32)
        self.value_fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten for dense layers
        flat = x.view(-1, 64 * 3 * 3)
        
        policy_logits = self.policy_fc(flat)
        
        v = F.relu(self.value_fc1(flat))
        value = torch.tanh(self.value_fc2(v))
        
        return policy_logits, value