import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Class to handle hardware architecture
class DeviceType:
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

    @staticmethod
    def get_best_device():
        if torch.cuda.is_available():
            return DeviceType.CUDA
        elif torch.backends.mps.is_available():
            return DeviceType.MPS
        else:
            return DeviceType.CPU
   
# Deep learning model for prediction of well data once a new well is discovered
class Model():
    """
    Description:
    Highly cusomisable pytorch neural network
    """
    def __init__(self, network_meta: list[dict]) -> None:
        """
        Description:
        Initalise the model and corresponding layers based on the passed in network meta
        
        params:
        network_meta (list[dict]): The shape of the neural network eg.
        [
            {
                'neurons': 2 # input layer / input dimension
            },
            {
                'neurons': 30,
                'activation': nn.ReLU,
                'type': nn.Linear
            },
            ...
            {
                'neurons': 1,
                'activation': nn.ReLU,
                'type': nn.Linear
            }
        ]
        """
        # List to store layer information
        self.layers = []

        self.network_meta = network_meta
        self.layer_count = len(self.network_meta)
                
        # Create the layers based on the network meta shape
        for i in range(1, self.layer_count):
            self.layers.append(
                # eg. nn.Linear
                self.network_meta[i]['type'](
                    # eg. 10
                    self.network_meta[i - 1]['neurons'],
                    # eg. 40
                    self.network_meta[i]['neurons']
                )
            )
            if self.network_meta[i]['activation'] is not None:
                # eg. nn.ReLU
                self.layers.append(self.network_meta[i]['activation']())
        
        # Create the model as sequential, unpacking layer data
        self.model = nn.Sequential(*self.layers)
        
        # Find the best device and move the model there
        self.device = DeviceType.get_best_device()
        print(f"Using device: {self.device}")
        if self.device != DeviceType.CPU:
            self.model = self.model.to(self.device)
    
    
    # Generic x and y data cleaner method
    def data_to_tensor(self, x, y):
        """
        Description:
        Change data from a numpy ndarray to a torch tensor
        """
        if type(x) == type(np.ndarray([])):
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x
            
        if type(y) == type(np.ndarray([])):
            y_tensor = torch.from_numpy(y).float()
        else:
            y_tensor = y
            
        return x_tensor, y_tensor
    
    
    def data_to_device(self, x, y):
        """
        Description:
        Add datasets to GPU if there is one
        """
        # Potemtially assert that the data is the correct type (torch tensor)
        if self.device != DeviceType.CPU:
            return x.to(self.device), y.to(self.device)
        else:
            return x, y
    
    
    def train(self,  x_train, y_train, n_epochs=500, learning_rate=1e-3, loss_fn=nn.MSELoss()) -> None:
        """
        Description:
        Train the model on the training data
        """
        
        # Clean up data and ensure that it is a torch tensor
        x_train_tensor, y_train_tensor = self.data_to_tensor(x_train, y_train)
        
        # Move data to the CPU if there is one
        x_train_tensor, y_train_tensor = self.data_to_device(x_train_tensor, y_train_tensor)
                
        # Adam optimisatoin
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialises the model to start training
        self.model.train()
        
        # Training loop
        for epoch in range(n_epochs):
            # Zero the gradients
            self.optimizer.zero_grad()  
            
            # Forward pass
            predictions = self.model(x_train_tensor)
            
            # Compute loss
            loss = loss_fn(predictions, y_train_tensor)
            
            # Backward pass, gradient descent
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Print the loss every 50 epochs for monitoring
            # if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.6f}")
    
    def test(self, x_test, y_test) -> None:
        """
        Test the model on training data
        """
        print("Testing data...")
        X_test_tensor, y_test_tensor = self.data_to_tensor(x_test, y_test)
        X_test_tensor, y_test_tensor = self.data_to_device(x_test, y_test)
        
        # Initialise the model to be evaluating
        self.model.eval()
        with torch.no_grad():
            test_predictions = self.model(X_test_tensor)
            mse = nn.MSELoss()(test_predictions, y_test_tensor).item()
        print(mse)
        return self
    
    # What to print when an instance is passed into print()
    def __str__(self):
        """
        Description:
        Magic method to print a summary of the model's architecture
        """
        model_summary = []
        model_summary.append(f"Model architecture:")
        for i, layer in enumerate(self.layers):
            model_summary.append(f"Layer {i}: {layer}")
        return "\n".join(model_summary)
    

# Delete this once the model has been created
if __name__ == "__main__":
    network_meta = [
        {
            'neurons': 2 # input layer / input dimension
        },
        {
            'neurons': 30,
            'activation': nn.ReLU,
            'type': nn.Linear
        },
        {
            'neurons': 50,
            'activation': nn.ReLU,
            'type': nn.Linear
        },
        {
            'neurons': 50,
            'activation': nn.ReLU,
            'type': nn.Linear
        },
        {
            'neurons': 1,
            'activation': None,
            # 'activation': nn.ReLU,
            'type': nn.Linear
        },
    ]
    
    m = Model(network_meta)
    
    x_train = torch.FloatTensor([[1,2], [3,4]])
    y_train = torch.FloatTensor([6, 4])
    
    train = True
    test = True
    
    if train:
        m.train(
            x_train=x_train,
            y_train=y_train
        )

    if test:
        m.test(
            x_test=x_train,
            y_test=y_train
        )
    
    print(m)