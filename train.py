""" Main training script for the model. """

import torch

from dataset import SyntheticDataset
from model import NeRF

def train(
    train_dataset_path: str = "",
    valid_dataset_path: str = "",
    test_dataset_path: str = "",
    run_name: str = "test",
    num_epochs: int = 10,
    num_layers: int = 3,
    num_hidden: int = 256,
    lr: float = 0.001,
    batch_size: int = 2,
) -> float:

    # TODO: Batch size is per-ray

    # Load the dataset
    train_dataset = SyntheticDataset(root=train_dataset_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    # Load the validation dataset
    valid_dataset = SyntheticDataset(root=valid_dataset_path)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)

    # Load the test dataset
    test_dataset = SyntheticDataset(root=test_dataset_path)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # Create the model
    model = NeRF(
        num_layers = num_layers,
        num_hidden = num_hidden,
    )

    # Create an Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create the loss function

    #TODO: Difference between ground truth pixel value and 
    #      sum of predicted pixel value is the loss
    loss_fn

    # Keep track of best loss so far
    best_loss = float("inf")

    for epoch in range(num_epochs):
        
        # Iterate over the dataset
        for image, theta, phi in train_dataloader:

            # Zero the gradients
            optimizer.zero_grad()

            # TODO: Actually do Rendering

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = loss_fn(outputs, labels)

            # Check to see if loss is the best so far
            if loss < best_loss:
                best_loss = loss

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

        # Validation
        for image, theta, phi in valid_dataloader:

            pass
    
    # Test
    for image, theta, phi in test_dataloader:

        pass

    return best_loss