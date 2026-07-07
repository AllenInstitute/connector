from __future__ import print_function
import argparse

import time
import torch
import utilities
import numpy as np
import sys
import math
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

import os
import copy

# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: 
        return torch.sigmoid(x) * x

class MyCustomDataset(Dataset):
    def __init__(self, data, obs_dim):
        try:
            self.rates = data['rates'][:,:,:obs_dim].astype(np.float32)
        except(KeyError):
            self.rates = data['spikes'][:,:,:obs_dim].astype(np.float32)
        try:
            self.externalinputs = data['externalinputs'].astype(np.float32)
        except(KeyError):
            self.externalinputs = np.full_like(self.rates[:,:,:obs_dim], np.nan, dtype=float)
        try:
            self.h0s = data['potentials'][:,0,:obs_dim].astype(np.float32)
        except(KeyError):
            self.h0s = np.full_like(self.rates[:,0,:obs_dim], np.nan, dtype=float)
        try:
            self.z0s = data['latents'][:,0,:].astype(np.float32)
        except(KeyError):
            self.z0s = np.full_like(self.rates[:,0,:], np.nan, dtype=float)
        try:
            self.mask = torch.arange(self.rates.shape[1]).expand(self.rates.shape[0], self.rates.shape[1]) >= data['lengths'].unsqueeze(1)
        except(KeyError):
            self.mask = np.full_like(self.z0s, np.nan, dtype=float)

    def __len__(self):
        return self.rates.shape[0]

    def __getitem__(self, idx):
        return self.rates[idx,:,:], self.externalinputs[idx,:,:], self.h0s[idx,:], self.z0s[idx,:], self.mask[idx,:]

class MyCustomLatentsDataset(Dataset):
    def __init__(self, data):
        self.latents = data['latents'].astype(np.float32)
        self.z0s = data['latents'][:,0,:].astype(np.float32)

        try:
            self.externalinputs = data['externalinputs'].astype(np.float32)
        except(KeyError):
            self.externalinputs = np.full_like(self.latents, np.nan, dtype=float)

        try:
            self.mask = torch.arange(self.latents.shape[1]).expand(self.latents.shape[0], self.latents.shape[1]) >= data['lengths'].unsqueeze(1)
        except(KeyError):
            self.mask = np.full_like(self.z0s, np.nan, dtype=float)

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx):
        return self.latents[idx,:,:], self.externalinputs[idx,:,:], self.z0s[idx,:], self.mask[idx,:]

class WeightsDataset(Dataset):
    def __init__(self, data):
        self.MBD = data.astype(np.float32)

    def __len__(self):
        return self.MBD.shape[0]

    def __getitem__(self, idx):

        return self.MBD[idx,:]

class LatentsDataset(Dataset):
    def __init__(self, data):
        self.latents = data['latents'].astype(np.float32)
        try:
            self.externalinputs = data['externalinputs'].astype(np.float32)
        except(KeyError):
            self.externalinputs = np.full_like(data['latents'], np.nan, dtype=float)

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx):
        return self.latents[idx,:,:], self.externalinputs[idx,:,:]

class ConnectivityDistribution(nn.Module):
    def __init__(self, mbd_dim: int = 2, n_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128, scale: float = 1.0):
        super().__init__()

        self.mbd_dim = mbd_dim
        self.n_dim = n_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.scale = scale # scaling between m and n

        self.mbd_model = nn.Sequential(
            nn.Linear(mbd_dim+time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, mbd_dim),
        )

        self.n_model = nn.Sequential(
            nn.Linear(n_dim+mbd_dim+time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, n_dim),
        )

    def mbd_forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.mbd_dim)
        t = t.reshape(-1, self.time_dim).float()
        
        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        mbd = self.mbd_model(torch.cat([x, t], dim=1))

        return mbd.reshape(*sz)
    
    def n_forward(self, x: Tensor, c: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.n_dim)
        t = t.reshape(-1, self.time_dim).float()
        c = c.reshape(-1, self.mbd_dim)

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        n = self.n_model(torch.cat([x, c, t], dim=1))
        
        return n.reshape(*sz)
    
    def m_sample(
        self, 
        num_samples: int,
        device='cuda',
        num_steps=100
    ):
        """
        Generate samples conditioned on a specific class.
        
        Args:
            model: Trained conditional velocity model
            num_samples: Number of samples to generate
            device: Device to run on
            dim: Dimensionality of samples
            num_steps: Number of ODE integration steps
        
        Returns:
            Generated samples [num_samples, dim]
        """
        
        with torch.no_grad():
            # Initial noise
            x_0 = torch.randn((num_samples, self.mbd_dim), dtype=torch.float32, device=device)
            
            # Wrap model to include conditioning
            def velocity(x, t):
                return self.mbd_forward(x, t)
            
            wrapped_model = ModelWrapper(velocity)
            T = torch.linspace(0,1,num_steps)  # sample times
            T = T.to(device=device)

            # Create solver and generate samples
            solver = ODESolver(velocity_model=wrapped_model)
            traj = solver.sample(
                time_grid=T, 
                x_init=x_0, 
                method='midpoint', 
                step_size=0.05, 
                return_intermediates=True
            )  # sample from the model
            
            # Return final samples
            return traj
    
    def n_sample(
        self, 
        num_samples: int, 
        context: float,
        device='cuda',
        num_steps=100
    ):
        """
        Generate samples conditioned on a specific class.
        
        Args:
            model: Trained conditional velocity model
            num_samples: Number of samples to generate
            context: Conditioning context
            device: Device to run on
            dim: Dimensionality of samples
            num_steps: Number of ODE integration steps
        
        Returns:
            Generated samples [num_samples, dim]
        """
        
        with torch.no_grad():
            # Initial noise
            x_0 = torch.randn((num_samples, self.n_dim), dtype=torch.float32, device=device)
            
            # Create condition labels (all same class)
            c = Tensor(context).to(device)
            c = c.view(-1, 1)
            
            # Wrap model to include conditioning
            def conditional_velocity(x, t):
                return self.n_forward(x, c, t)
            
            wrapped_model = ModelWrapper(conditional_velocity)
            T = torch.linspace(0,1,num_steps)  # sample times
            T = T.to(device=device)

            # Create solver and generate samples
            solver = ODESolver(velocity_model=wrapped_model)
            traj = solver.sample(
                time_grid=T, 
                x_init=x_0, 
                method='midpoint', 
                step_size=0.05, 
                return_intermediates=True
            )  # sample from the model
            
            # Return final samples
            return traj
        
    def sample(
        self,
        nsamples: int,
        device='cuda',
        sigma=0.0
    ) -> Tensor:
        mbd = self.m_sample(
            num_samples=nsamples,
            device=device,
            num_steps=10
        )[-1,:,:]
        n = self.n_sample(
            num_samples=nsamples,
            context=mbd,
            device=device,
            num_steps=10
        )[-1,:,:] * self.scale
        n += sigma*torch.randn_like(n)
        return mbd, n
    
def train(args):
    # set random seed for reproducibility
    torch.manual_seed(args.seed)

    # set device based on accelerator availability
    if args.accel and not torch.accelerator.is_available():
        print("ERROR: accelerator is not available, try running on CPU")
        sys.exit(1)
    if not args.accel and torch.accelerator.is_available():
        print("WARNING: accelerator is available, run with --accel to enable it")

    if args.accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    kwargs = {'num_workers': 1, 'pin_memory': True} if device=="cuda" else {}

    # Loading and preprocessing the data
    data = np.load(args.datapath, allow_pickle=True)
    if args.obs_dim == -1:
        args.obs_dim = data['MBD'].shape[0]
    train_indices, valid_indices, test_indices = utilities.split_data(data['latents'], args.n_splits, args.seed)
    weights_data = WeightsDataset(data['MBD'][:args.obs_dim,:])
    latents_data = LatentsDataset(data)
    train_subset = Subset(latents_data, train_indices[args.k-1])
    valid_subset = Subset(latents_data, valid_indices[args.k-1])
    test_subset = Subset(latents_data, test_indices[args.k-1])

    weights_loader = DataLoader(
        weights_data,
        batch_size=args.batch_size,
        shuffle=True, 
        **kwargs
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=data['latents'].shape[0],
        shuffle=True, 
        **kwargs
    )

    valid_loader = DataLoader(
        valid_subset,
        batch_size=data['latents'].shape[0],
        shuffle=True,
        **kwargs
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=data['latents'].shape[0],
        shuffle=True, 
        **kwargs
    )

    activation_map = {
        'tanh': F.tanh,
        'relu': F.relu,
        'softplus': F.softplus,
        'sigmoid': F.sigmoid,
    }

    activation = activation_map[args.activation]

    externalinput_dim = data['externalinputs'].shape[-1] if 'externalinputs' in data else 0

    model = ConnectivityDistribution(
        mbd_dim=data['MBD'].shape[-1],
        n_dim=args.latent_dim,
        time_dim=1,
        hidden_dim=args.hidden_dim,
        scale=args.scale
    ).to(device) 

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    # instantiate an affine path object
    path = AffineProbPath(scheduler=CondOTScheduler())

    def train_mbd_epoch(epoch, optimizer):
        print_every = 100
        model.train()

        # Freeze all model parameters
        for param in model.parameters():
            param.requires_grad = False
        
        for param in model.mbd_model.parameters():
            param.requires_grad = True
        
        losses = []
        for batch in weights_loader:
            # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
            x_1 = batch.to(device)
            x_0 = torch.randn_like(x_1).to(device)

            # sample time (user's responsibility)
            t = torch.rand(x_1.shape[0]).to(device)

            # sample probability path
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

            # flow matching l2 loss
            loss = torch.pow(
                model.mbd_forward(path_sample.x_t,path_sample.t) - \
                path_sample.dx_t,
                2
            ).mean()

            # optimizer step
            loss.backward() # backward
            optimizer.step() # update
            optimizer.zero_grad() 
            losses.append(loss.detach())
        
        losses = torch.stack(losses)
        # log loss
        if (epoch+1) % print_every == 0:
            print('| iter {:6d} | loss {:8.5f}' 
                .format(epoch+1, losses.mean().item()))

        return losses.mean().item()
    
    def train_n_epoch(epoch, optimizer):
        print_every = 100
        model.train()

        # Freeze all model parameters
        for param in model.parameters():
            param.requires_grad = False
        
        for param in model.n_model.parameters():
            param.requires_grad = True
        
        train_loss = 0
        for batch_idx, (zs, externalinputs) in enumerate(train_loader):
            zs = zs.to(device)
            optimizer.zero_grad()

            sol = model.m_sample(
                num_samples=args.nsamples,
                device=device,
                num_steps=10
            ) 
            MBD_model = sol[-1,:,:].to(device)
            M_model = MBD_model[:, :args.latent_dim]
            if args.bias:
                D_model = MBD_model[:,-1]
            else:
                D_model = 0

            Y = zs[:,1:,:] + (args.alpha-1)*zs[:,:-1,:]
            Y = Y.reshape(-1, Y.shape[-1])

            if not torch.isnan(externalinputs).any():
                externalinputs = externalinputs.to(device)
                if args.bias:
                    B_model = MBD_model[:, args.latent_dim:-1]
                else:
                    B_model = MBD_model[:, args.latent_dim:]
                X = activation(zs[:, :-1, :] @ M_model.T + externalinputs[:, :-1, :] @ B_model.T + D_model)
            else:
                X = activation(zs[:,:-1,:] @ M_model.T + D_model)
            
            X = X.reshape(-1, X.shape[-1])
            reg = ((args.regularizer*(args.nsamples**2))/(args.alpha**2))*torch.eye(X.shape[1])
            N_hat = args.nsamples*torch.linalg.solve(X.T @ X + reg.to(device), X.T @ Y)/args.alpha

            x_1 = N_hat.to(device)/args.scale
            x_0 = torch.randn_like(x_1).to(device)

            # sample time (user's responsibility)
            t = torch.rand(x_1.shape[0]).to(device) 

            # sample probability path
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

            # flow matching l2 loss
            loss = torch.pow(
                model.n_forward(path_sample.x_t,MBD_model,path_sample.t) - \
                path_sample.dx_t, 
                2
            ).mean() 
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # log loss
        if (epoch+1) % print_every == 0:
            print('| iter {:6d} | loss {:8.5f}' 
                .format(epoch+1, train_loss / len(train_loader.dataset)))

        return train_loss / len(train_loader.dataset)

    def valid_n_epoch(epoch):
        model.eval()

        valid_loss = 0
        for batch_idx, (zs, externalinputs) in enumerate(valid_loader):
            zs = zs.to(device)

            sol = model.m_sample(
                num_samples=args.nsamples,
                device=device,
                num_steps=10
            ) 
            MBD_model = sol[-1,:,:].to(device)
            M_model = MBD_model[:, :args.latent_dim]
            if args.bias:
                D_model = MBD_model[:,-1]
            else:
                D_model = 0

            Y = zs[:,1:,:] + (args.alpha-1)*zs[:,:-1,:]
            Y = Y.reshape(-1, Y.shape[-1])

            if not torch.isnan(externalinputs).any():
                externalinputs = externalinputs.to(device)
                if args.bias:
                    B_model = MBD_model[:, args.latent_dim:-1]
                else:
                    B_model = MBD_model[:, args.latent_dim:]
                X = activation(zs[:, :-1, :] @ M_model.T + externalinputs[:, :-1, :] @ B_model.T + D_model)
            else:
                X = activation(zs[:,:-1,:] @ M_model.T + D_model)
            
            X = X.reshape(-1, X.shape[-1])
            reg = ((args.regularizer*(args.nsamples**2))/(args.alpha**2))*torch.eye(X.shape[1])
            N_hat = args.nsamples*torch.linalg.solve(X.T @ X + reg.to(device), X.T @ Y)/args.alpha

            x_1 = N_hat.to(device)/args.scale
            x_0 = torch.randn_like(x_1).to(device)

            # sample time (user's responsibility)
            t = torch.rand(x_1.shape[0]).to(device) 

            # sample probability path
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

            # flow matching l2 loss
            loss = torch.pow(
                model.n_forward(path_sample.x_t,MBD_model,path_sample.t) - \
                path_sample.dx_t, 
                2
            ).mean() 

            valid_loss += loss.item()

        return valid_loss / len(valid_loader.dataset)

    def test_n_epoch(epoch):
        model.eval()

        test_loss = 0
        for batch_idx, (zs, externalinputs) in enumerate(test_loader):
            zs = zs.to(device)

            sol = model.m_sample(
                num_samples=args.nsamples,
                device=device,
                num_steps=10
            ) 
            MBD_model = sol[-1,:,:].to(device)
            M_model = MBD_model[:, :args.latent_dim]
            if args.bias:
                D_model = MBD_model[:,-1]
            else:
                D_model = 0

            Y = zs[:,1:,:] + (args.alpha-1)*zs[:,:-1,:]
            Y = Y.reshape(-1, Y.shape[-1])

            if not torch.isnan(externalinputs).any():
                externalinputs = externalinputs.to(device)
                if args.bias:
                    B_model = MBD_model[:, args.latent_dim:-1]
                else:
                    B_model = MBD_model[:, args.latent_dim:]
                X = activation(zs[:, :-1, :] @ M_model.T + externalinputs[:, :-1, :] @ B_model.T + D_model)
            else:
                X = activation(zs[:,:-1,:] @ M_model.T + D_model)
            
            X = X.reshape(-1, X.shape[-1])
            reg = ((args.regularizer*(args.nsamples**2))/(args.alpha**2))*torch.eye(X.shape[1])
            N_hat = args.nsamples*torch.linalg.solve(X.T @ X + reg.to(device), X.T @ Y)/args.alpha

            x_1 = N_hat.to(device)/args.scale
            x_0 = torch.randn_like(x_1).to(device)

            # sample time (user's responsibility)
            t = torch.rand(x_1.shape[0]).to(device) 

            # sample probability path
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

            # flow matching l2 loss
            loss = torch.pow(
                model.n_forward(path_sample.x_t,MBD_model,path_sample.t) - \
                path_sample.dx_t, 
                2
            ).mean() 

            test_loss += loss.item()

        return test_loss / len(test_loader.dataset)
    
    train_mbd_losses = []
    train_n_losses = []
    valid_n_losses = []
    test_n_losses = []

    best_val_loss = float("inf")
    best_model_state = None
    
    print('Inferring connectivity distribution over m, b, d...')
    for epoch in range(1, args.mbd_training_epochs+1):
        train_loss = train_mbd_epoch(epoch, optimizer)
        train_mbd_losses.append(train_loss)

    print('Inferring connectivity distribution over n given m, b, d...')
    for epoch in range(1, args.n_training_epochs+1):
        train_loss = train_n_epoch(epoch, optimizer)
        valid_loss = valid_n_epoch(epoch)
        test_loss = test_n_epoch(epoch)
        train_n_losses.append(train_loss)

        ## Save the best model based on validation loss
        #if valid_loss < best_val_loss:
        #    best_val_loss = valid_loss
        #    best_model_state = model.state_dict()

        valid_n_losses.append(valid_loss)
        test_n_losses.append(test_loss)

    # Save the best model (lowest validation loss)
    # if best_model_state is not None:
    checkpoint = {
        "model_state_dict": copy.deepcopy(model.state_dict()), #model.state_dict(), #best_model_state,
        "args": vars(args),  # convert argparse.Namespace to dict
        "best_val_loss": -1, # best_val_loss,
        "train_mbd_losses": train_mbd_losses,
        "train_n_losses": train_n_losses,
        "valid_n_losses": valid_n_losses,
        "test_n_losses": test_n_losses,
    }

    # Create directory if it doesn't exist
    os.makedirs(args.modelpath, exist_ok=True)
    torch.save(checkpoint, f"{args.modelpath}/model_fold_{args.k}_seed_{args.seed}_id_{args.train_id}.pth")
    #print(f"Best model (val_loss={best_val_loss:.4f}) saved to {args.modelpath}/model_fold_{args.k}_seed_{args.seed}_id_{args.train_id}.pth")
    print("Training complete.")
    return model, train_mbd_losses, train_n_losses, valid_n_losses, test_n_losses

##### Custom code for LINT (Valente et al.)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=2, generator=None)
        #nn.init.uniform_(m.weight, a=-3.0, b=3.0, generator=None)
        #nn.init.orthogonal_(m.weight, gain=10, generator=None)
        #nn.init.xavier_uniform_(m.weight, gain=1e-3, generator=None)
        if m.bias is not None:
            nn.init.normal_(m.bias, mean=0.0, std=1e-3, generator=None)

class ValenteRNN(nn.Module):
    def __init__(
            self, 
            latent_dim,
            hidden_dim,
            externalinput_dim,
            activation=F.tanh,
            alpha=1.0,
            bias=False
        ):
        super(ValenteRNN, self).__init__()

        # Initialize parameters
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.externalinput_dim = externalinput_dim
        self.activation = activation
        self.alpha = alpha
        
        # Define the neural network architecture
        self.h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.linear1 = nn.Linear(latent_dim + externalinput_dim, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, latent_dim, bias=False)

        self.apply(init_weights)

    def flow(self, z, v):
        if torch.isnan(v).any():
            h = self.activation(self.linear1(z))
            z_dot = -z + self.linear2(h)/self.hidden_dim
        else:
            h = self.activation(self.linear1(torch.cat((z, v), dim=-1)))  # Concatenate z and v
            z_dot = -z + self.linear2(h)/self.hidden_dim
        return z_dot

    def forward(self, h, u):
        # forward difference
        z = self.linear2(self.activation(h))
        if torch.isnan(u).any():
            h_dot = -h + self.linear1(z)/self.hidden_dim
        else:
            h_dot = -h + self.linear1(self.activation(torch.cat((z, u), dim=-1)))/self.hidden_dim
        h_new = h + self.alpha * h_dot
        return h_new, h_dot
    
    def init_hidden(self, h0):
        if h0 is not None:
            return h0
        return self.h0

def train_lint(args):
    # set random seed for reproducibility
    torch.manual_seed(args.seed)

    # set device based on accelerator availability
    if args.accel and not torch.accelerator.is_available():
        print("ERROR: accelerator is not available, try running on CPU")
        sys.exit(1)
    if not args.accel and torch.accelerator.is_available():
        print("WARNING: accelerator is available, run with --accel to enable it")

    if args.accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    kwargs = {'num_workers': 1, 'pin_memory': True} if device=="cuda" else {}

    # Loading and preprocessing the data
    data = np.load(args.datapath, allow_pickle=True)
    if args.obs_dim == -1:
        args.obs_dim = data['rates'].shape[-1]
    train_indices, valid_indices, test_indices = utilities.split_data(data['rates'], args.n_splits, args.seed)
    dataset = MyCustomDataset(data, args.obs_dim)
    train_subset = Subset(dataset, train_indices[args.k-1])
    valid_subset = Subset(dataset, valid_indices[args.k-1])
    test_subset = Subset(dataset, test_indices[args.k-1])

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True, 
        **kwargs
    )

    valid_loader = DataLoader(
        valid_subset,
        batch_size=data['rates'].shape[0],
        shuffle=True, 
        **kwargs
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=data['rates'].shape[0],
        shuffle=True, 
        **kwargs
    )

    activation_map = {
        'tanh': F.tanh,
        'relu': F.relu,
        'softplus': F.softplus,
        'sigmoid': F.sigmoid,
    }

    externalinput_dim = data['externalinputs'].shape[-1] if 'externalinputs' in data else 0
    activation = activation_map[args.activation]
    model = ValenteRNN(
        latent_dim=args.latent_dim,
        hidden_dim=args.obs_dim,
        externalinput_dim=externalinput_dim,
        activation=activation,
        alpha=args.alpha,
        bias=args.bias
    ).to(device) 

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr
    )

    loss_fun = nn.MSELoss()

    def train_epoch(epoch, optimizer):
        model.train()
        train_loss = 0
        for batch_idx, (rates, externalinputs, h0s, _, _) in enumerate(train_loader):
            rates = rates.to(device)
            if not torch.isnan(externalinputs).any():
                externalinputs = externalinputs.to(device)
            optimizer.zero_grad()
            
            hstates = torch.zeros_like(rates, device=device)
            if not torch.isnan(h0s).any():
                hstate = h0s.to(device)
            else:
                hstate = model.init_hidden(None).expand(rates.shape[0], -1)

            hstates[:,0,:] = hstate
            for i in range(1, rates.shape[1]):
                h_new, h_dot = model.forward(
                    hstate,
                    externalinputs[:,i-1,:]
                )
                hstates[:,i,:] = h_new
                hstate = h_new

            # Manually calculate L2 regularization term
            l2_reg = torch.tensor(0., requires_grad=True)
            for param in model.parameters():
                l2_reg = l2_reg + torch.norm(param, 2) # L2 norm

            loss = loss_fun(
                activation(hstates),
                rates
            ) + args.regularizer*l2_reg

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        return train_loss / len(train_loader.dataset)

    def valid_epoch(epoch):
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for _, (rates, externalinputs, h0s, _, _) in enumerate(valid_loader):
                rates = rates.to(device)
                if not torch.isnan(externalinputs).any():
                    externalinputs = externalinputs.to(device)
                
                hstates = torch.zeros_like(rates, device=device)
                if not torch.isnan(h0s).any():
                    hstate = h0s.to(device)
                else:
                    hstate = model.init_hidden(None).expand(rates.shape[0], -1)

                hstates[:,0,:] = hstate
                for i in range(1, rates.shape[1]):
                    h_new, h_dot = model.forward(
                        hstate,
                        externalinputs[:,i-1,:]
                    )
                    hstates[:,i,:] = h_new
                    hstate = h_new

                loss = loss_fun(
                    activation(hstates),
                    rates
                )
                
                valid_loss += loss.item()

        return valid_loss / len(valid_loader.dataset)

    def test_epoch(epoch):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for _, (rates, externalinputs, h0s, _, _) in enumerate(test_loader):
                rates = rates.to(device)
                if not torch.isnan(externalinputs).any():
                    externalinputs = externalinputs.to(device)
                
                hstates = torch.zeros_like(rates, device=device)
                if not torch.isnan(h0s).any():
                    hstate = h0s.to(device)
                else:
                    hstate = model.init_hidden(None).expand(rates.shape[0], -1)

                hstates[:,0,:] = hstate
                for i in range(1, rates.shape[1]):
                    h_new, h_dot = model.forward(
                        hstate,
                        externalinputs[:,i-1,:]
                    )
                    hstates[:,i,:] = h_new
                    hstate = h_new

                loss = loss_fun(
                    activation(hstates),
                    rates
                )

                test_loss += loss.item()

        return test_loss / len(test_loader.dataset)

    train_losses = []
    valid_losses = []
    test_losses = []
    total_epochs = args.standard_training_epochs

    best_val_loss = float("inf")
    best_model_state = None
    
    for epoch in range(1, total_epochs + 1): # Can we select parameter set that does best on validation set?
        train_loss = train_epoch(epoch, optimizer)
        valid_loss = valid_epoch(epoch)
        test_loss = test_epoch(epoch)
        print('| epoch {:6d} | train loss {:8.5f} | val loss {:8.5f} | test loss {:8.5f}  (x 1000)'.format(
            epoch, 1000*train_loss, 1000*valid_loss, 1000*test_loss))
        train_losses.append(train_loss)

        # Save the best model based on validation loss
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_model_state = copy.deepcopy(model.state_dict())

        valid_losses.append(valid_loss)
        test_losses.append(test_loss)

    # Save the best model (lowest validation loss)
    if best_model_state is not None:
        checkpoint = {
            "model_state_dict": best_model_state,
            "args": vars(args),  # convert argparse.Namespace to dict
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "test_losses": test_losses,
        }

        # Create directory if it doesn't exist
        os.makedirs(args.modelpath, exist_ok=True)
        torch.save(checkpoint, f"{args.modelpath}/model_fold_{args.k}_seed_{args.seed}_id_{args.train_id}.pth")
        print(f"Best model (val_loss={best_val_loss:.4f}) saved to {args.modelpath}/model_fold_{args.k}_seed_{args.seed}_id_{args.train_id}.pth")
        print("Training complete.")
        model.load_state_dict(best_model_state)
        return model, train_losses, valid_losses, test_losses
    else:
        print("Training complete.")
        return model, train_losses, valid_losses, test_losses