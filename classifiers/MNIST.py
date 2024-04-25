from torch import nn, Tensor
import torch.nn.functional as F



class Classifier(nn.Module):
    """
    Simple model with linear layers for mnist
    """

    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.drop2 = nn.Dropout(p=0.5)
        self.out = nn.Linear(256, 10)
        self.out_act = nn.Softmax(dim=1)


    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.out(x)
        x = self.out_act(x)
        return x

    #def __init__(self):
        #super(Classifier, self).__init__()
        #self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        #self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        #self.drop1 = nn.Dropout2d(p=0.5)
        #self.fc1 = nn.Linear(9216, 128)
        #self.drop2 = nn.Dropout2d(p=0.5)
        #self.fc2 = nn.Linear(128, 10)
        
    #def forward(self, x):
        #x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        #x = self.max_pool(x)
        #x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        #x = self.drop1(x)
        #x = F.relu(self.fc1(x))
        #x = self.drop2(x)
        #x = self.fc2(x)
        #return x        
