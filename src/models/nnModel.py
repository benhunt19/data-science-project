import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from src.models.modelClassBase import Model
   
# Deep learning model for prediction of well data once a new well is discovered
class NNModel(Model):
    """
    Description:
    Highly cusomisable pytorch neural network!
    """
    def __init__(self, network_meta: list[dict], useGPU : bool = False) -> None:
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
        super().__init__()
        # List to store layer information
        self.layers = []
        self.useGPU = useGPU

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
        self.device = DeviceType.get_best_device(useGPU)
        print(f"Using device: {self.device}")
        if self.device != DeviceType.CPU:
            self.model = self.model.to(self.device)

    def data_to_tensor(self, x, y) -> tuple:
        """
        Description:
            Convert input data to PyTorch tensors.
        
        Parameters:
            x: Input features (DataFrame, ndarray, or tensor)
            y: Target values (ndarray or tensor)
        
        Returns:
            tuple: (x_tensor, y_tensor) as PyTorch float tensors
        """
        if isinstance(x, (pd.DataFrame, pd.Series, np.ndarray)):
            x_tensor = torch.from_numpy(np.asarray(x)).float()
        elif isinstance(x, torch.Tensor):
            x_tensor = x.float()
        else:
            raise TypeError(f"Unsupported input type for x: {type(x)}")

        if isinstance(y, (np.ndarray, pd.Series, pd.DataFrame)):
            y_tensor = torch.from_numpy(np.asarray(y)).float()
        elif isinstance(y, torch.Tensor):
            y_tensor = y.float()
        else:
            raise TypeError(f"Unsupported input type for y: {type(y)}")

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
    
    def train(self,  x_train, y_train, n_epochs=10, learning_rate=0.01, loss_fn=nn.MSELoss(), optimizer=optim.Adam) -> None:
        """
        Description:
        Train the model on the training data
        """
        
        # Clean up data and ensure that it is a torch tensor
        x_train_tensor, y_train_tensor = self.data_to_tensor(x_train, y_train)
        
        # Move data to the CPU if there is one and enabled
        x_train_tensor, y_train_tensor = self.data_to_device(x_train_tensor, y_train_tensor)
                
        data_handler = DataSetHandler(
            x_train_tensor,
            y_train_tensor
        )
                
        # Sefine optimizer
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(n_epochs):
            # Zero the gradients
            
            # Initialises the model to start training
            self.model.train()

            # Create a dataloader from the initial data handler
            loader = DataLoader(
                data_handler,
                batch_size=2**10,
                shuffle=True,          # Randomize order each epoch
                num_workers=1,         # Parallel loading
                # pin_memory=True        # Faster CPU-to-GPU transfer
            )

            for batch_X, batch_y in loader:
                
                # Move each batch to GPU if enabled
                batch_X, batch_y = self.data_to_device(batch_X, batch_y)
                self.optimizer.zero_grad()  
                
                # Forward pass
                predictions = self.model(batch_X)
                
                # Compute loss
                loss = loss_fn(predictions, batch_y)
                
                # Backward pass, gradient descent
                loss.backward()
                
                # Update weights
                self.optimizer.step()
            
                # Move each batch to GPU
                # batch_data = batch_data.to(device)
                # batch_targets = batch_targets.to(device)
                
                # optimizer.zero_grad()
                # output = model(batch_data)
                # loss = criterion(output, batch_targets)
                # loss.backward()
                # optimizer.step()
            
            # Print the loss every 10 epochs for monitoring
            if (epoch + 1) % 1 == 0 or epoch == 0:
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
 
    
# Class to handle dataset
class DataSetHandler(Dataset):
    def __init__(self, X, y):
        self.X = X      # Features (e.g., tensor of shape [n_samples, n_features])
        self.y = y      # Labels (e.g., tensor of shape [n_samples])

    def __len__(self):
        return len(self.X)  # Number of samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] # Return tuple (data, label)

# Class to handle hardware architecture
class DeviceType:
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

    @staticmethod
    def get_best_device(useGPU):
        if torch.cuda.is_available() and useGPU:
            return DeviceType.CUDA
        elif torch.backends.mps.is_available() and useGPU:
            return DeviceType.MPS
        else:
            return DeviceType.CPU



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
    
    m = NNModel(network_meta)
    
    x_train = torch.FloatTensor([[1.3, 2.6], [3.1, 4.7], [5.8, 6.9]])
    y_train = torch.FloatTensor([6, 4, 2])
    
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