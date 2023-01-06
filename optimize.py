"""Optimize Hyperparameters for MLP/Data using HyperOpt."""

import numpy as np
import hyperopt
from hyperopt import fmin, hp, tpe

from train import train

# Define the objective function
def objective(params):

  # The run name will be used to create a folder for the run
  run_name = f"run_{params['num_layers']}_{params['num_hidden']}_{params['batch_size']}_{params['lr']}"

  # Perform the run
  best_loss = train(
    num_epochs = params['num_epochs'],
    num_layers = params['num_layers'],
    num_hidden = params['num_hidden'],
    lr = params['lr'],
    batch_size = params['batch_size'],
    run_name = run_name,
  )
  return best_loss


if __name__ == "__main__":

    # Define the search space
    search_space = {
        'batch_size': hp.choice('batch_size', [128, 256, 512]),
        'lr': hp.loguniform('lr',  np.log(0.0001), np.log(0.01)),
        'num_epochs': 200,
        'num_layers': hp.choice('num_layers' [1, 2, 3]),
        'num_hidden': hp.choice('num_hidden' [128, 256]),
    }

    # Run the optimization
    best = fmin(objective, space=search_space, algo=tpe.suggest, max_evals=50)

    # Print the best dataset found
    print(best)