"""Optimize Hyperparameters for MLP/Data using HyperOpt."""

import hyperopt
from hyperopt import fmin, hp, tpe

from optimize import perform_one_run

# Define the objective function
def objective(params):

  # The run name will be used to create a folder for the run
  run_name = f"run_{params['encoder']}_{params['batch_size']}_{params['lr']}"

  # Perform the run
  best_loss = perform_one_run(
    run_name = run_name,
    num_epochs=params["num_epochs"],
    step_size=params["step_size"],
    gamma=params["gamma"],
    lr = params["lr"],
    batch_size = params["batch_size"],
    tren_csv_file = f"train_{params['encoder']}.csv",
    test_csv_file = f"test_{params['encoder']}.csv",
    pred_csv_file = f"pred_{run_name}.csv",
  )
  return best_loss


if __name__ == "__main__":

    # Define the search space
    search_space = {
        'encoder': hp.choice('encoder', [
          'esm1v_t33_650M_UR90S_1',
          'esm1v_t33_650M_UR90S_5',
          'esm2_t33_650M_UR50D',
        ]),
        'batch_size': hp.choice('batch_size', [128, 256, 512]),
        'lr': hp.loguniform('lr',  np.log(0.0001), np.log(0.01)),
        'num_epochs': 200,
        # Learning rate scheduler
        'step_size': 50,
        'gamma': 0.1,
    }

    # Run the optimization
    best = fmin(objective, space=search_space, algo=tpe.suggest, max_evals=50)

    # Print the best dataset found
    print(best)