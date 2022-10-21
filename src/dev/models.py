import torch
import torch.nn as nn
import torch.nn.functional as F
from config import IMG_WIDTH, IMG_HEIGHT

class CTC(nn.Module):
    def __init__(
        self, vocab_size: int,
        input_size: int = 64,
        hidden_size: int = 32,
        dropout: float = 0.25
        ):
        super(CTC,self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size = (3,5), padding = (1,2))
        self.pool = nn.MaxPool2d(kernel_size = (2,2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size = (3,5), padding = (1,2))
        self.dropout = nn.Dropout(p = dropout)
        self.linear = nn.Linear(1152, input_size)
        self.gru = nn.GRU(input_size, hidden_size, bidirectional = True, num_layers = 2,
                          dropout = dropout, batch_first = True)
        self.output = nn.Linear(hidden_size * 2, vocab_size + 1)
        self.logsoftmax = nn.LogSoftmax(dim = 2)
        
    def forward(self, inputs):
        bs, _, _, _ = inputs.size()
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)
        x = F.relu(self.linear(x))
        x = self.dropout(x)
        x, _ = self.gru(x)
        x = self.output(x)
        out = self.logsoftmax(x)
        return out ### [batch_size, input_length, num_classes (including blank token)]
    
if __name__ == "__main__":
    model = CTC(19)
    sample = torch.rand(5, 3, IMG_HEIGHT, IMG_WIDTH)
    output = model(sample)
    print(output.shape)