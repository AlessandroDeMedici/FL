# seaborn confusion matrix colorata

# rivedere frase che classical machine learning non efficiente

# fare confronto fra binario n-ario


#introduzione
    # 6g ai
    #background strumenti usati
        # FL
        # autoencoder
# ambiente






from collections import OrderedDict

import flwr as fl
from flwr.common import Metrics
from flwr_datasets import FederatedDataset

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import hickle as hkl
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np


import torchvision.transforms as transforms
tensor_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=90),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  
    transforms.RandomHorizontalFlip(),  
])


# definizione di una classe per la parte di autoencoding
class Autoencoder(nn.Module):
    def __init__(self, z_dim = 16):
        super(Autoencoder, self).__init__()

        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=4, stride=1, padding=0, groups=1),  # depthwise
            nn.Conv2d(1, 64, kernel_size=1),  # pointwise
            nn.LeakyReLU(True),

            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=0, groups=64),  # depthwise
            nn.Conv2d(64, 128, kernel_size=1),  # pointwise
            nn.LeakyReLU(True),

            nn.Flatten(0),

            nn.Linear(2048, 512),  # output 512
            nn.LeakyReLU(True),
            nn.Linear(512, z_dim),  # output z_dim
            nn.Sigmoid()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.LeakyReLU(True),
            nn.Linear(512, 2048),

            nn.LeakyReLU(True),
            nn.Unflatten(0, (128, 4, 4)),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=1, padding=0)
        )
        

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

    # funzione che ritorna la rappresentazione di x nello spazio latente
    def latent(self, x):
        return self.encoder(x)


def train(model, train_loader, num_epochs, device):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    # learning rate molto piu' piccolo                                            
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = data
            inputs = tensor_transforms(inputs)
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # Usiamo gli input originali senza rumore
            loss = criterion(outputs, inputs)  # Confrontiamo con gli input originali
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print('Finished Training')
    return avg_loss


def test(model, test_loader, device):
    criterion = torch.nn.MSELoss()
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data  # Assuming labels are available
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            test_loss += loss.item()
            total += labels.size(0)
    
    avg_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.4f}')
    return avg_loss

def get_latent(model, test_loader, device):
    latent_rep = []
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = model.latent(inputs)
            outputs = np.array(outputs.cpu())
            latent_rep.append(outputs)
    
    latent_rep = np.array(latent_rep)
    return latent_rep

def predict(model, test_loader, device):
    recon_rep = []
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = np.array(outputs.cpu())
            recon_rep.append(outputs)
    
    recon_rep = np.array(recon_rep)
    return recon_rep

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
