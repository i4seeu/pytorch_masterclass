## understanding tensor
import torch

# Create a tensor
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x)

##operations

# Addition
y = torch.tensor([[7, 8, 9], [10, 11, 12]])
print(x + y)

# Element-wise multiplication
print(x * y)

# Matrix multiplication
print(torch.matmul(x, y.T))  # Transpose y to match dimensions

# Create a tensor with requires_grad=True to track computation for gradients
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = 2 * x + 3

# Compute gradients
y.backward(torch.tensor([1.0, 1.0]))  # The gradient of y with respect to x

# Access gradients
print(x.grad)  # Will print [2, 2]


## Neaural Networks
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 5)  # Fully connected layer with input size 10 and output size 5
    
    def forward(self, x):
        x = self.fc(x)
        return x

# Create an instance of the network
net = SimpleNet()
