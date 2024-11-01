import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, OrderedDict

# definizione di una classe per la parte di autoencoding
class Autoencoder(nn.Module):
    def __init__(self, z_dim = 16):
        super(Autoencoder, self).__init__()

        self.z_dim = z_dim

        # Encoder
        # self.encoder = nn.Sequential(
        #     nn.Linear(100,1000),
        #     nn.LeakyReLU(True),
        #     nn.Linear(1000,64),
        #     nn.LeakyReLU(True),
        #     nn.Linear(64,32),
        #     nn.LeakyReLU(True)
        # )

        # self.fc_mu = nn.Linear(32, z_dim)
        # self.fc_log_var = nn.Linear(32, z_dim)
        
        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.LeakyReLU(True),
        #     nn.Linear(z_dim, 32),
        #     nn.LeakyReLU(True),
        #     nn.Linear(32,64),
        #     nn.LeakyReLU(True),
        #     nn.Linear(64,512),
        #     nn.LeakyReLU(True),
        #     nn.Linear(512,100),
        #     nn.LeakyReLU(True)
        # )

        self.encoder = nn.Sequential(
            # 3x3 conv
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=1, padding=0), # 7x7x16
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=0), # 4x4x32
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0), # 2x2x64
            nn.LeakyReLU(True),
            # 1x1 conv
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0), # 2x2x128
            nn.LeakyReLU(True),
            nn.Flatten(0)
        )

        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_log_var = nn.Linear(512, z_dim)
        
        # # Decoder
        self.decoder = nn.Sequential(
            nn.LeakyReLU(True),
            nn.Linear(16,512),
            nn.Unflatten(0, (128,2,2)),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0), # 64x4x4
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=0), # 32x7x7
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=1, padding=0), # 16x10x10
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(in_channels=16,out_channels=1, kernel_size=1, stride=1, padding=0) # 1x10x10
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
        y = self.decoder(z).view(10,10)
        return y, mu, log_var

    # funzione che ritorna la rappresentazione di x nello spazio latente
    def latent(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        z = self.reparameterize(mu,log_var)
        return z


def vae_loss(recon_x, x, mu, log_var, KL_WEIGHT, training=True):
    # Loss di ricostruzione
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # Divergenza KL
    kl_div = -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp())

    # Loss contrattiva
    if training:
        # Assicurarsi che x richieda il gradiente
        if not x.requires_grad:
            x.requires_grad = True

        # Loss contrattiva
        grad_mu = torch.autograd.grad(mu.sum(), x, retain_graph=True, create_graph=True)[0]
        contractive_loss = grad_mu.pow(2).sum()
    else:
        contractive_loss = 0

    # Loss totale
    total_loss = recon_loss #+KL_WEIGHT * kl_div + contractive_loss

    return total_loss, recon_loss, kl_div


KL_WEIGHT = 1 / (12 * 12)
def train(model, train_loader, num_epochs, device):
    global KL_WEIGHT

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6) # lower learning rate
    model.train()  # Set the model to training mode

    for epochs in range(num_epochs):
        train_loss = 0
        mean_loss = 0
        kl_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            data.requires_grad = True
            optimizer.zero_grad() 
            recon_batch, mu, log_var = model(data)
            loss,ml,kl = vae_loss(recon_batch, data, mu, log_var, KL_WEIGHT)

            loss.backward() 
            train_loss += loss.item()
            mean_loss += ml.item()
            kl_loss += kl.item()
            optimizer.step()

        average_loss = train_loss / len(train_loader.dataset)
        avg_mean_loss = mean_loss / len(train_loader.dataset)
        avg_kl_loss = kl_loss / len(train_loader.dataset)
        print(f'Epoch:[{epochs + 1}/{num_epochs}] Train loss: {average_loss}, Recon loss: {avg_mean_loss}, KL Divergence: {avg_kl_loss}')

        if epochs >= num_epochs/2:
            KL_WEIGHT = 0
        else:
            KL_WEIGHT = KL_WEIGHT / 2

    return average_loss


def test(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    with torch.no_grad():  # Disable gradient computation for testing (saves memory and computations)
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            KL_WEIGHT = 1
            loss,_,_= vae_loss(recon_batch, data, mu, log_var, KL_WEIGHT, training=False)
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