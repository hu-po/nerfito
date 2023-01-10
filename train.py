""" Main training script for the model. """

import torch

from dataset import SyntheticDataset
from model import NeRF
from geometry import Camera, Volume, get_points

def train(
    train_dataset_path: str = "",
    valid_dataset_path: str = "",
    test_dataset_path: str = "",
    run_name: str = "test",
    num_epochs: int = 10,
    num_layers: int = 3,
    num_hidden: int = 256,
    use_leaky_relu: bool = False,
    lr: float = 0.001,
    batch_size: int = 2,
) -> float:

    # Create the volume object
    volume = Volume(

    )

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
        num_layers=num_layers,
        num_hidden=num_hidden,
        use_leaky_relu=use_leaky_relu,
    )

    # Create an Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Keep track of best loss so far
    best_loss = float("inf")

    for epoch in range(num_epochs):

        # Iterate over the dataset
        for d in train_dataloader:

            # Zero the gradients
            optimizer.zero_grad()

            # Create the camera object
            camera = Camera(
                position=d["camera_position"],
                orientation=d["camera_orientation"],
                focal_length=0.5,
                img_width=128,
                img_height=128,
            )

            # Extract theta and phi from datapoint dict
            theta = d["theta"]
            phi = d["phi"]

            # Get all the points that are sampled for this ray
            for point in get_points(camera, volume):
                # Combine the theta, phi, and sample point locations into a input tensor
                single_point_model_input = torch.cat((theta, phi, point), dim=1)

            # Forward pass (N x 4)
            outputs = model(single_point_model_input)

            # Each batch is going to represent a single ray
            assert outputs.shape[0] == batch_size

            # Compute the final pixel value for each output ray (ray marching)
            predicted_pixel_value = torch.zeros(batch_size, 3)
            for point in range(batch_size):
                opacity = outputs[point, 0]
                color = outputs[point, 1:]
                # TODO: Once a certain opacity threshold has been reached, there should be
                # no more color added to the pixel value
                predicted_pixel_value += color * (1 - opacity)

            # Compute the loss
            loss = torch.mean((predicted_pixel_value - d["pixel_value"])**2)

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
