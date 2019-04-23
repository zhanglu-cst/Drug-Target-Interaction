from torch import nn


class NN(nn.Module):
    def __init__(self,input_size = 64*2,n_h1 = 500,n_h2 = 500):
        super().__init__()
        self.layer1 = nn.Sequential(
                nn.Linear(input_size,n_h1),
                nn.BatchNorm1d(n_h1),
                nn.ReLU(),
                nn.Dropout(),
        )
        self.layer2 = nn.Sequential(
                nn.Linear(n_h1, n_h2),
                nn.BatchNorm1d(n_h2),
                nn.ReLU(),
                nn.Dropout(),
        )
        self.layer3 = nn.Sequential(
                nn.Linear(n_h2, 1),
                nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x