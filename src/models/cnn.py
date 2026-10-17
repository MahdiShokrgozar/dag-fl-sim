import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    """
    A standard CNN architecture for MNIST dataset.
    Input: (Batch, 1, 28, 28)
    Output: (Batch, 10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class CIFARCNN(nn.Module):
    """
    Same depth/capacity as MNISTCNN (2x conv+pool, 2x fc) - kept comparable on
    purpose, sized only for CIFAR-10's 3x32x32 input (vs MNIST's 1x28x28), so
    a CIFAR-10 run isolates the dataset/resolution change rather than also
    changing model capacity (see thesis-hub/docs/decisions.md: Farabi-Prox is
    flagged as untested on larger models like CIFAR/ResNet - this class is
    deliberately NOT that test).
    Input: (Batch, 3, 32, 32)
    Output: (Batch, 10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class CIFARCNNGN(nn.Module):
    """
    Deeper CIFAR-10 CNN with GroupNorm (2026-09-24, thesis-hub/docs/decisions.md
    same date). Plan 38 reached 55% held-out with CIFARCNN under the best
    FARABI config; the student set a >70% target. CIFARCNN deliberately mirrors
    MNISTCNN's depth, so this is an opt-in alternative, not a replacement.

    Roughly the same parameter count as CIFARCNN (~0.55M: CIFARCNN spends
    most of its parameters in fc1, this one in five 3x3 conv layers), so
    model transfer size and aggregation cost stay comparable - the difference
    is depth and normalization, not size.

    GroupNorm, not BatchNorm: BatchNorm's running statistics are skewed by
    non-IID local data and do not average meaningfully across clients
    (Hsieh et al., ICML 2020, "The Non-IID Data Quagmire"); GroupNorm has no
    running buffers, so every state_dict entry is an ordinary float parameter.
    Input: (Batch, 3, 32, 32)
    Output: (Batch, 10)
    """
    def __init__(self, groups: int = 8):
        super().__init__()

        def block(c_in, c_out):
            return [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(groups, c_out),
                nn.ReLU(inplace=True),
            ]

        self.features = nn.Sequential(
            *block(3, 64), *block(64, 64), nn.MaxPool2d(2),       # 16x16
            *block(64, 128), *block(128, 128), nn.MaxPool2d(2),   # 8x8
            *block(128, 256), nn.MaxPool2d(2),                    # 4x4
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
        x = self.dropout(x)
        return self.fc(x)
