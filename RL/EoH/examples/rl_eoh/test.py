def __init__(self, output_size, channels=16, kernel_size=3):\n
super(ConvBase, self).__init__()\n
self.conv1 = nn.Conv1d(1, channels, kernel_size, stride=2, padding=1)\n
self.bn1 = nn.BatchNorm1d(channels)\n
self.conv2 = nn.Conv1d(channels, channels * 2, kernel_size, stride=2, padding=1)\n
self.bn2 = nn.BatchNorm1d(channels * 2)\n
self.conv3 = nn.Conv1d(channels * 2, channels * 4, kernel_size, stride=2, padding=1)\n
self.bn3 = nn.BatchNorm1d(channels * 4)\n        self.dropout = nn.Dropout(0.1)\n
# Compute flattened size: input 40 → after conv1: floor((40+2*1-3)/2)+1 = 20 → conv2: 10 → conv3: 5\n
# self.linear_in = channels * 4 * 5\n        self.linear = nn.Linear(self.linear_in, output_size)\n    def forward(self, x, verbose=False):\n        x = x.view(-1, 1, x.shape[-1])\n        h = self.conv1(x)\n        h = self.bn1(h).relu()\n        h = self.conv2(h)\n        h = self.bn2(h).relu()\n        h = self.conv3(h)\n        h = self.bn3(h).relu()\n        h = self.dropout(h)\n        h = h.view(h.shape[0], -1)\n        return logits', 'objective': np.float64(-9.999999999999999e+17