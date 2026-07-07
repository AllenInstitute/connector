from __future__ import print_function
import argparse
import train

# Default parameters
parser = argparse.ArgumentParser(
    description='Connector'
)
parser.add_argument(
    '--k', type=int, default=1,
    help='index of the current fold (default: 1)'
)
parser.add_argument(
    '--n_splits', type=int, default=5,
    help='number of splits for KFold cross-validation (default: 5)'
)
parser.add_argument(
    '--latent_dim', type=int, default=2,
    help='dimensionality of the latent space (default: 2)'
)
parser.add_argument(
    '--hidden_dim', type=int, default=500,
    help='dimensionality of the hidden layer (default: 500)'
)
parser.add_argument(
    '--obs_dim', type=int, default=-1,
    help='dimensionality of the observation space (default: -1)'
)
parser.add_argument(
    '--batch_size', type=int, default=25,
    help='input batch size for training (default: 25)'
)
parser.add_argument(
    '--accel', action='store_true', 
    help='use accelerator'
)
parser.add_argument(
    '--seed', type=int, default=1,
    help='random seed (default: 1)'
)
parser.add_argument(
    '--lr', type=float, default=1e-4,
    help='learning rate (default: 1e-4)'
)
parser.add_argument(
    '--weight_decay', type=float, default=1e-5,
    help='weight decay (default: 1e-5)'
)
parser.add_argument(
    '--activation', type=str, default='tanh',
    choices=['tanh', 'relu', 'softplus', 'sigmoid'],
    help='activation function (default: tanh)'
)
parser.add_argument(
    '--scale', type=float, default=5.0,
    help='scale factor for the input (default: 5.0)'
)
parser.add_argument(
    '--standard_training_epochs', type=int, default=1000, # 1000 for quad, 1000 for others
    help='number of standard training epochs (default: 1000)'
)
parser.add_argument(
    '--mbd_training_epochs', type=int, default=100,
    help='number of epochs for mbd training (default: 100)'
)
parser.add_argument(
    '--n_training_epochs', type=int, default=100,
    help='number of epochs for n training (default: 100)'
)
parser.add_argument(
    '--regularizer', type=float, default=1.0, # higher regularization gives more weight on the prior N(0, I/regularizer)
    help='regularization strength (default: 1.0)'
)
parser.add_argument(
    '--nsamples', type=int, default=500,
    help='number of samples or "neurons" used for inference (default: 500)'
)
parser.add_argument(
    '--datapath', type=str, default='../quadstable_attractors.npz',
    help='path to the data file'
)
parser.add_argument(
    '--modelpath', type=str, default='../quadstable_attractors',
    help='path to the model file'
)
parser.add_argument(
    '--alpha', type=float, default=1.0,
    help='alpha parameter (default: 1.0)'
)

parser.add_argument(
    '--train_id', type=int, default=0,
    help='ID of the training run (default: 0)'
)
args = parser.parse_args()

if __name__ == "__main__":
    train.train(args)