import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Deep learning model for prediction of well data once a new well is discovered
class Model(nn.Module):
    # def __init__(self, network_shape: list[int]) -> None:
    def __init__(self) -> None:
        super().__init__()  # Initialize parent class nn.Module
        
        # if not isinstance(network_shape, (list, tuple)) or len(network_shape) < 2:
            # raise ValueError("network_shape must be a list/tuple with at least 2 elements")
            
        # self.network_shape = network_shape

        self.network_shape = [2, 20, 30, 40, 1]
        self.layers = nn.ModuleList()
        
        # Create the layers based on the network shape
        for i in range(len(self.network_shape) - 1):
            self.layers.append(nn.Linear(self.network_shape[i], self.network_shape[i + 1]))
            self.layers.append(nn.ReLU())
        
    def forward(self, x):
        # Define the forward pass of the model here
        for layer in self.layers:
            x = layer(x)
        return x
    
    # def train(self, train_loader, criterion, optimizer, num_epochs=1, device='cpu'):
        
    #     for epoch in range(num_epochs):
    #         for i, (inputs, labels) in enumerate(train_loader):
    #             # Forward pass
    #             outputs = self(inputs)
    #             loss = criterion(outputs, labels)
                
    #             # Backward and optimize
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    def __str__(self):
        return f"Model, shape: {self.networkShape}"  # Change this to return the name of the model
    

# Delete this once the model has been created
if __name__ == "__main__":
    m = Model()
    test_tesnsor = torch.FloatTensor([[1,2], [3,4]])
    test_res = torch.FloatTensor([6, 4])
    m(test_tesnsor)