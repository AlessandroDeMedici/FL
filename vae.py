import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, OrderedDict

# definizione di una classe per la parte di autoencoding
class Autoencoder(nn.Module):
    def __init__(self, z_dim = 12):
        super(Autoencoder, self).__init__()

        self.z_dim = z_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),  
            nn.LeakyReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),  
            nn.LeakyReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),  
            nn.LeakyReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),  
            nn.LeakyReLU(True),
            nn.Flatten(0),
            nn.LeakyReLU(True)
        )

        self.fc_mu = nn.Linear(256*2*2, z_dim)
        self.fc_log_var = nn.Linear(256*2*2, z_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.LeakyReLU(True),
            nn.Linear(z_dim, 256 * 2 * 2),
            nn.Unflatten(0,(256,2,2)),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(True)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    # funzione che ritorna la rappresentazione di x nello spazio latente
    def latent(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        return self.reparameterize(mu,log_var)


def vae_loss(recon_x, x, mu, log_var, KL_WEIGHT):
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x,x, reduction='mean')

    # KL divergence
    kl_div = -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp())

    # Total loss
    total_loss = recon_loss + KL_WEIGHT * kl_div

    return total_loss, recon_loss, kl_div




def train(model, train_loader, num_epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # lower learning rate
    model.train()  # Set the model to training mode

    for epochs in range(num_epochs):
        train_loss = 0
        mean_loss = 0
        kl_loss = 0

        KL_WEIGHT = epochs / (num_epochs * 64 * 64)

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()  # Zero out the gradients to avoid accumulation
            recon_batch, mu, log_var = model(data)
            
            
            loss,ml,kl = vae_loss(recon_batch, data, mu, log_var, KL_WEIGHT)
            
            loss.backward()  # Compute the gradients
            train_loss += loss.item()
            mean_loss += ml.item()
            kl_loss += kl.item()
            optimizer.step()  # Update the model parameters

        average_loss = train_loss / len(train_loader.dataset)
        avg_mean_loss = mean_loss / len(train_loader.dataset)
        avg_kl_loss = kl_loss / len(train_loader.dataset)
        print(f'Epoch:[{epochs + 1}/{num_epochs}] Train loss: {average_loss}, Recon loss: {avg_mean_loss}, KL Divergence: {avg_kl_loss}')

    return average_loss


def test(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    with torch.no_grad():  # Disable gradient computation for testing (saves memory and computations)
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            KL_WEIGHT = 1
            loss,_,_= vae_loss(recon_batch, data, mu, log_var, KL_WEIGHT)
            test_loss += loss.item()
    average_loss = test_loss / len(test_loader.dataset)
    print(f'Average test loss: {average_loss}')
    return average_loss


def get_latent(model, test_loader, device):
    latent_rep = []
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            output = model.latent(inputs)
            latent_rep.append(output.cpu())
    
    latent_rep = np.array(latent_rep)
    return latent_rep

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def predict(model, test_loader, device):
    recon_rep = []
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            outputs,_,_ = model(inputs)
            outputs = np.array(outputs.cpu())
            recon_rep.append(outputs)
    
    recon_rep = np.array(recon_rep)
    return recon_rep

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def model_init(model):
    model.apply(weights_init)