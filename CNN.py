import torch
from torch import nn

batch_size = 16
word_embed_size = 4
seq_len = 7
input = torch.randn(batch_size, word_embed_size, seq_len)
conv1 = nn.Conv2d(in_channels=word_embed_size, out_channels=3, kernel_size=3, stride=1, padding=1)
hidden1 = conv1(input)
hidden2 = torch.max(hidden1, dim=2)  # max pool

print(hidden1)
print(hidden2)
