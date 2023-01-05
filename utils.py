""" Ray and Volume utilities. """

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

# The up direction in world space
WORLD_UP_VEC: np.ndarray = np.array([0, 0, 1], dtype=np.float32)

@dataclass
class Ray:
    """Dataclass for a ray object."""
    origin: np.ndarray
    direction: np.ndarray

@dataclass
class Camera:
    """Dataclass for a camera object."""
    position: np.ndarray
    orientation: np.ndarray
    image_width: int
    image_height: int
    focal_length: float
    forward_direction_vector: np.ndarray = np.array([0, 0, -1], dtype=np.float32)
    right_direction_vector: np.ndarray = np.array([0, 1, 0], dtype=np.float32)

@dataclass
class Volume:
    """Dataclass for a volume object."""
    size: np.ndarray
    resolution: np.ndarray
    position: np.ndarray = np.array([0, 0, 0])


def get_rays(camera: Camera) -> List[Ray]:
    """Get a list of rays for a given camera."""

    # Calculate the camera's forward direction
    cam_forward = np.dot(camera.orientation, camera.forward_direction_vector)

    # Calculate the image plane's center point in world space
    image_center = camera.position + cam_forward * camera.focal_length

    # Calculate the image plane's width and height in world space
    image_width = 2 * camera.focal_length * np.tan(np.radians(camera.image_width / 2))
    image_height = 2 * camera.focal_length * np.tan(np.radians(camera.image_height / 2))

    # Calculate the image plane's right and up vectors in world space
    cam_right = np.cross(cam_forward, camera.right_direction_vector.T)
    cam_up = np.cross(cam_forward, cam_right)

    # Calculate the image plane's top left corner point in world space
    image_top_left = image_center + cam_up * (image_height / 2) - cam_right * (image_width / 2)

    # Calculate the step sizes for the right and up vectors
    right_step = cam_right * (image_width / camera.image_width)
    up_step = cam_up * (image_height / camera.image_height)

    # Initialize a list to store the rays
    rays = []

    # Loop through each pixel and calculate the ray
    for y in range(camera.image_height):
        for x in range(camera.image_width):

            # Calculate the pixel's position in world space
            pixel_pos = image_top_left + right_step * x + up_step * y

            # Calculate the ray's origin and direction
            ray_origin = camera.position
            ray_direction = pixel_pos - camera.position

            # Normalize the ray direction
            ray_direction /= np.linalg.norm(ray_direction)

            # Add the ray to the list
            rays.append(Ray(origin=ray_origin, direction=ray_direction))

    return rays

# ray sampling

# volume sampling

# volume rendering

if __name__ == "__main__":
    
    # Debugging code for the ray calculation
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the aspect ratio to equal
    ax.set_aspect('equal')

    # Create a camera object
    camera = Camera(
        position=np.array([0, -2, 2], dtype=np.float32),
        orientation=np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.6744415163993835, -0.7383282780647278],
        [0.0, 0.7383282780647278, 0.6744415163993835],
        ], dtype=np.float32),
        image_width=28,
        image_height=28,
        focal_length=0.5,
    )

    # Get the rays for the camera
    rays = get_rays(camera)

    # Plot the rays as lines
    for ray in get_rays(camera):
        ax.plot(
            [ray.origin[0], ray.direction[0]], 
            [ray.origin[1], ray.direction[1]],
            [ray.origin[2], ray.direction[2]],
        '-o')

    # Set the axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()

