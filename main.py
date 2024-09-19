import argparse

from autoencoder import Autoencoder, train, test, get_latent, get_parameters, set_parameters, predict

import hickle as hi

from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering, Birch
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader, TensorDataset, random_split, IterableDataset, Dataset, ConcatDataset

import flwr as fl
from flwr.common import Parameters, Scalar, FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation


from typing import Dict, List, Optional, Tuple, Union, OrderedDict
from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner


# libraries for metrics
from sklearn.metrics import f1_score, confusion_matrix, pairwise_distances
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import RobustScaler

# confusion matrix
import seaborn as sns

# transformer
import torchvision.transforms as transforms
tensor_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=90),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  
    transforms.RandomHorizontalFlip(),  
])



DEVICE = torch.device("mps")
disable_progress_bar()

# Global parameters
NUM_ELEMENTS = 2000
NUM_CLUSTER = 10
NUM_CLIENTS = 2
BATCH_SIZE = 1
NUM_EPOCHS = 30
NUM_ROUNDS = 5
SERVER_ID = 8

# load data
data = hi.load('data/data.hkl')
x_test = data['xtest']
y_test = data['ytest']
x_train = data['xtrain']
y_train = data['ytrain']

indici = y_train == 0
x_train = x_train[indici]
y_train = y_train[indici]

trainsets = []
valsets = []
testsets = []

# Function to partition data
def partition_data():
    global x_test, x_train, y_test, y_train, trainsets, valsets, testsets

    X_train = torch.tensor(x_train, dtype=torch.float32)
    Y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(x_test, dtype=torch.float32)
    Y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create TensorDatasets
    trainset = TensorDataset(X_train, Y_train)
    testset = TensorDataset(X_test, Y_test)
    valset = testset

    traingen = torch.Generator().manual_seed(42)
    testgen = torch.Generator().manual_seed(42)
    valgen = torch.Generator().manual_seed(42)

    trainlen = [NUM_ELEMENTS] * (int(len(x_train)/NUM_ELEMENTS)) + [len(x_train) % NUM_ELEMENTS]
    testlen = [NUM_ELEMENTS] * (int(len(x_test)/NUM_ELEMENTS)) + [len(x_test) % NUM_ELEMENTS]
    vallen = [NUM_ELEMENTS] * (int(len(x_test)/NUM_ELEMENTS)) + [len(x_test) % NUM_ELEMENTS]

    # split sets in partitions in a deterministic manner
    trainsets = random_split(trainset, trainlen, traingen)
    testsets = random_split(testset, testlen, testgen)
    valsets = random_split(valset, vallen, valgen)


# Function to load partition of data
def load_datasets(partition_id: int):
    global trainsets, testsets, valsets

    # Create data loaders with the custom dataset
    trainloader = DataLoader(trainsets, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(valsets[partition_id], batch_size=BATCH_SIZE, shuffle=False)
    testloader = DataLoader(testsets[partition_id], batch_size=BATCH_SIZE, shuffle=False)


    trainloader = DataLoader(trainsets[partition_id], batch_size=BATCH_SIZE, shuffle=False)
    valloader = DataLoader(valsets[partition_id], batch_size=BATCH_SIZE, shuffle=False)
    testloader = DataLoader(testsets[partition_id], batch_size=BATCH_SIZE, shuffle=False)

    return trainloader, valloader, testloader


# class for Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(model=self.net, train_loader=self.trainloader, num_epochs=NUM_EPOCHS, device=DEVICE)

        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss =  test(model=self.net, test_loader=self.valloader, device=DEVICE)
        accuracy = 0
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    
# Client function
def client_fn(cid: str) -> FlowerClient:

    # Load model
    model = Autoencoder()
    model.load_state_dict(torch.load('data/model.pth'))
    net = model.to(DEVICE)
    
    # Load data
    partition_id = int(cid)
    trainloader, valloader, _ = load_datasets(partition_id=partition_id)

    for data in trainloader:
        x,y = data
        x = tensor_transforms(x)

    return FlowerClient(net, trainloader, valloader).to_client()


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        net = Autoencoder()

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(net.state_dict(), "data/model.pth")

        return aggregated_parameters, aggregated_metrics

# Create FedAvg strategy
strategy = SaveModelStrategy(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=NUM_CLIENTS,  # Never sample less than 5 clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
)



# define client resources
client_resources = {"num_cpus": 1, "num_gpus": 0.0}

# parsing function
def parsing():
    global NUM_EPOCHS, NUM_CLIENTS, NUM_ROUNDS, y_test, y_train, NUM_CLUSTER, NUM_ELEMENTS, SERVER_ID
    parser = argparse.ArgumentParser(description="Processing inputs")

    parser.add_argument('--train', action='store_true', required=False, help='Train the model')
    parser.add_argument('--tsne', action='store_true', help='Plot 2D data with T-SNE')
    parser.add_argument('--epochs', type=int, required=False, help='Number of epochs of training for each client')
    parser.add_argument('--rounds', type=int, required=False, help='Number of rounds of Federated Learning')
    parser.add_argument('--clients', type=int, required=False, help='Number of clients of Federated Learning')
    parser.add_argument('--binary', action='store_true', help='Perform a binary classification test')
    parser.add_argument('--clear', action='store_true', help='Train a new model')
    parser.add_argument('--plot', action='store_true', help='Plot reconstructed images')
    parser.add_argument('--classify', action='store_true', help='Plot reconstructed images')
    parser.add_argument('--elements', type=int, required=False, help='Number of elements per client')
    parser.add_argument('--server', type=int, required=False, help='Server partition ID')
    parser.add_argument('--save', action='store_true', help='Save latent and labels')
    parser.add_argument('--normal', type=int, nargs='+', required=False, help='Array of labels to set as normal class (0)')

    args = parser.parse_args()

    if args.clients is not None:
        NUM_CLIENTS = args.clients
    if args.epochs is not None:
        NUM_EPOCHS = args.epochs
    if args.rounds is not None:
        NUM_ROUNDS = args.rounds
    if args.binary:
        NUM_CLUSTER = 2
        y_test[y_test == 8] = 0
        y_test[y_test != 0] = 1
        y_train[y_train == 8] = 0
        y_train[y_train != 0] = 1
    if args.elements is not None:
        NUM_ELEMENTS = args.elements
    if args.server is not None:
        SERVER_ID = args.server
    if args.normal is not None:
        for label in args.normal:
            if label < 0 or label > 9:
                continue
            y_test[y_test == label] = 0
            NUM_CLUSTER = NUM_CLUSTER - 1

    if args.clear:
        model = Autoencoder()
        torch.save(model.state_dict(), 'data/model.pth')

    return args

def plot_sets(tensors, recons):
    num_images = 100
    grid_size = 10
    num_rows = 10

    # Create a figure with dynamic number of rows and a fixed number of columns
    fig, axs = plt.subplots(num_rows, grid_size, figsize=(grid_size, num_rows))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    i = 0
    for tensor in tensors:
        tensor = tensor[0]  # Assuming each tensor is wrapped in an extra dimension
        normalized_tensor = (tensor + 1) / 2
        numpy_tensor = normalized_tensor

        ax = axs[i // grid_size, i % grid_size]
        ax.imshow(numpy_tensor, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        i += 1
        if i >= num_images:  # Stop if the number of images is less than expected
            break

    # Second plot for reconstructed images
    fig, axs = plt.subplots(num_rows, grid_size, figsize=(grid_size, num_rows))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    i = 0
    for recon in recons:
        recon = recon[0]  # Assuming each recon is wrapped in an extra dimension
        normalized_recon = (recon + 1) / 2
        numpy_recon = normalized_recon

        ax = axs[i // grid_size, i % grid_size]
        ax.imshow(numpy_recon, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        i += 1
        if i >= num_images:
            break

    plt.show()




if __name__ == "__main__":

    # parsing of inputs
    args = parsing()

    # partition data
    partition_data()

    # define model
    model = Autoencoder()

    # train model
    if args.train is not None and args.train:

        # Start simulation
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
            client_resources=client_resources,
        )

    # load model
    try:
        model.load_state_dict(torch.load('data/model.pth'))
        net = model.to(DEVICE)
    except:
        print("Unable to load model - exiting")
        exit(1)


    # load data
    trainloader, valloader, testloader = load_datasets(partition_id=SERVER_ID)

    # Get latent representation of inputs
    latent_rep = get_latent(net, testloader, DEVICE)
    recon_rep = predict(net, testloader, DEVICE)
    
    # normalize data
    scaler = RobustScaler()
    latent_rep = scaler.fit_transform(latent_rep)

    if args.save is not None and args.save:
        y = testloader.dataset[:][1].numpy()
        print(latent_rep.shape)
        print(y)
        np.savez_compressed('x.npz', latent_rep)
        np.savez_compressed('y.npz', y)

    # CLASSIFICATION AND METRICS
    if args.classify is not None and args.classify:
        actual_labels = testloader.dataset[:][1]

        # Apply Gaussian Mixture Model
        gmm = GaussianMixture(n_components=NUM_CLUSTER)
        #km = KMeans(n_clusters=NUM_CLUSTER,random_state=17,init='k-means++',n_init=20,algorithm='elkan')
        # agg = AgglomerativeClustering(n_clusters=NUM_CLUSTER)
        # sc = SpectralClustering(n_components=NUM_CLUSTER)
        # bc = Birch(n_clusters=NUM_CLUSTER)
        # dbscan = DBSCAN(eps=0.5)
        y_predette = gmm.fit_predict(latent_rep)

        # confusion matrix
        conf_matrix = confusion_matrix(actual_labels, y_predette)
        # find optimal mapping with Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-conf_matrix)

        # create mapping between predicted clusters and true labels
        mapping = {col: row for row, col in zip(row_ind, col_ind)}

        # remap y_predette based on the optimal mapping
        y_predette_mapped = np.array([mapping[pred] for pred in y_predette])

        # confusion matrix after mapping
        conf_matrix_mapped = confusion_matrix(actual_labels, y_predette_mapped)

        plt.figure('Confusion Matrix',figsize=(8,8))
        sns.heatmap(conf_matrix_mapped, annot=True, fmt='d')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')

        # calculate F1 scores
        f1_scores_macro = f1_score(actual_labels, y_predette_mapped, average='macro')
        f1_scores_micro = f1_score(actual_labels, y_predette_mapped, average='micro')
        f1_scores_weighted = f1_score(actual_labels, y_predette_mapped, average='weighted')
        f1_scores = f1_score(actual_labels, y_predette_mapped, average=None)


        print(f"F1 scores: ({NUM_ELEMENTS} test-samples)")
        for i, f1 in enumerate(f1_scores):
            print(f'class: {i}: {f1}')
        
        # print how many elements for each class
        #unique, counts = np.unique(y_test, return_counts=True)
        #class_distribution = dict(zip(unique, counts))
        #print(class_distribution)

        print(f'f1 scores (macro averaging): {f1_scores_macro}')
        print(f'f1 scores (micro averaging): {f1_scores_micro}')
        print(f'f1 scores (weighted averaging): {f1_scores_weighted}')

    if args.tsne is not None and args.tsne:

        #tsne = TSNE(n_components=2, random_state=42, method='exact')
        #data_tsne = tsne.fit_transform(latent_rep)
        pca = PCA(n_components=2)
        data_tsne = pca.fit_transform(latent_rep)


        plt.figure('Predicted',figsize=(6, 6))
        scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=y_predette_mapped,cmap='winter' ,s=50, alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
        plt.title("Predicted")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)


        plt.figure('Actual',figsize=(6, 6))
        scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=testloader.dataset[:][1], cmap='winter' ,s=50, alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
        plt.title("Actual")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)

    if args.plot:
        plot_sets(testloader.dataset, recon_rep)


    plt.show()

    

