from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F

## Model 1- Setup + Basic Skeleton + Lighter Model
class MNIST_CNN_M1(nn.Module):
    def __init__(self):
        super(MNIST_CNN_M1, self).__init__()
        ## Input size: 28, Input RF: 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False), ## Output size: 26, RF: 3
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False), ## Output size: 24, RF: 5
            nn.ReLU(),
            ## Transition Block 1
            nn.MaxPool2d(2, 2), ## Output size 12, RF: 6
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False), ## Output size 12, RF: 6
            nn.ReLU()
        ) ## Output size 12, RF: 6, J:2


        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False), ## Output size 10, RF: 10
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False), ## Output size 8, RF: 14
            nn.ReLU(),
            ## Transition Block 2
            nn.MaxPool2d(2, 2), ## Output size 4, RF: 28
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False), ## Output size 4, RF: 28
            nn.ReLU()
        ) ## Output size 4, RF: 28, J:4

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(4, 4), padding=0, bias=False),
            nn.ReLU()
        ) ## RF: 28, J:4

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(-1,10)
        x = F.log_softmax(x, dim=1)
        return x

## Model 2- Batch Normalization+ Regularization+ Global Average Pooling
class MNIST_CNN_M2(nn.Module):
    def __init__(self):
        super(MNIST_CNN_M2, self).__init__()
        ## Input size: 28, Input RF: 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False), ## 26x26 RF: 3
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.ReLU(),

            ## Transition Block 1
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.ReLU(),

            ## Transition Block 2
            # nn.MaxPool2d(2, 2), 
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) ## Output size 4, RF: 16

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7)
        ) # output_size = 1


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gap(x)

        x = x.view(-1,10)
        x = F.log_softmax(x, dim=1)
        return x

## Model 3- Increasring Capacity+ Correcting MaxPooling + Playing Naively with Learning Rates
class MNIST_CNN_M3(nn.Module):
    def __init__(self):
        super(MNIST_CNN_M3, self).__init__()
        ## Input size: 28, Input RF: 1
        ## First Conv Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), padding=0, bias=False), ## 26x26 RF: 3
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Dropout(0.05),

            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), padding=0, bias=False), ## 24x24 RF: 5
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.05),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), ## 22x22 RF: 7
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), ## 20x20 RF: 9
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),

            ## Transition Block 1
            nn.MaxPool2d(2, 2), ## 10x10 RF: 18
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False) ## 10x10 RF: 18
            # nn.ReLU()
            # nn.BatchNorm2d(8),
            # nn.Dropout(0.1),
        ) ## Output size 10, RF: 18

        ## Second Conv Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), ## 8x8 RF: 22
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), ## 6x6 RF: 26
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
        ) 

        ## GAP  
        self.gap = nn.Sequential(   
            nn.AdaptiveAvgPool2d(1)  ## 1x1 RF: 26
        ) # output_size = 1

        ## Third Conv Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False) ## 1x1 RF: 26
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gap(x)
        x = self.conv3(x)

        x = x.view(-1,10)
        x = F.log_softmax(x, dim=1)
        return x


## Model 4- Final Model- Discipline
class MNIST_CNN_M4(nn.Module):
    def __init__(self):
        super(MNIST_CNN_M4, self).__init__()
        ## Input size: 28, Input RF: 1
        ## First Conv Block
        self.conv1 = nn.Sequential(
            ## Conv Layer 1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False), ## 26x26 RF: 3
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.05),

            ## Conv Layer 2
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), ## 24x24 RF: 5
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),

            ## Transition Block
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False), ## 24x24 RF: 5
            nn.MaxPool2d(2, 2) ## 12x12 RF: 10
        ) ## Output size 12, RF: 10

        ## Second Conv Block
        self.conv2 = nn.Sequential(
            ## Conv Layer 3
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False), ## 10x10 RF: 14
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05),

            ## Conv Layer 4
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), ## 8x8 RF: 18
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),

            ## Conv Layer 5
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), ## 6x6 RF: 22
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),

            ## Conv Layer 6
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, bias=False) ## 4x4 RF: 26
        ) 

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1) ## 1x1 RF: 26
        )

    def forward(self, x):
        ## First Conv Block
        x = self.conv1(x)

        ## Second Conv Block
        x = self.conv2(x)

        ## GAP
        x = self.gap(x)

        ## Flatten
        x = x.view(-1,10)

        ## Log Softmax
        x = F.log_softmax(x, dim=1)
        return x
