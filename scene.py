"""Synthetic Dataset Generation."""

import bpy
import math
import numpy as np

OBJECT_SIZE = 0.8
OBJECT_LOCATION = (0, 0, 0)
CAMERA_LOCATION = (0, 0, 0)
CAMERA_ROTATION = (0, 0, 0)
CAMERA_SPHERE_SAMPLES = 10
CAMERA_SPHERE_RADIUS = 4.0
CAMERA_SPHERE_LOCATION = (0, 0, 0)
CAMERA_IMAGE_SIZE = (56, 56)
IMAGE_FILEPATH = "/tmp/"

# Delete the default cube
bpy.ops.object.delete()

# Spawn the monkey head object
bpy.ops.mesh.primitive_monkey_add(size=OBJECT_SIZE, location=OBJECT_LOCATION)
monkey_head = bpy.data.objects["Suzanne"]

# Set the object color to purple
monkey_head.color = (1, 0, 1)  # RGB color (purple)

# Get the object called "Camera"
camera = bpy.data.objects["Camera"]

# Add a tracking constraint to the camera
tracking_constraint = camera.constraints.new(type='TRACK_TO')
tracking_constraint.target = monkey_head
tracking_constraint.track_axis = 'TRACK_NEGATIVE_Z'
tracking_constraint.up_axis = 'UP_Y'

# Set the camera's initial position and rotation
camera.location = CAMERA_LOCATION
camera.rotation_euler = CAMERA_ROTATION

# Print the 3x3 rotation matrix of the object
print(f'Camera rotation matrix is: {camera.matrix_world.to_3x3()}')

# Generate a meshgrid of points on the sphere
theta, phi = np.meshgrid(
  np.linspace(0, 2 * math.pi, CAMERA_SPHERE_SAMPLES),
  np.linspace(0, math.pi, CAMERA_SPHERE_SAMPLES),
)

# Iterate over the samples on the sphere
for theta, phi in zip(theta.flatten(), phi.flatten()):

  # Calculate the x, y, and z coordinates for this sample
  x = CAMERA_SPHERE_RADIUS * math.sin(theta) * math.cos(phi)
  y = CAMERA_SPHERE_RADIUS * math.sin(theta) * math.sin(phi)
  z = CAMERA_SPHERE_RADIUS * math.cos(theta)

  # Set the camera's position to the sample point
  print(f'Camera location is: ({x}, {y}, {z})')
  camera.location = (x, y, z)

  # bpy.ops.mesh.primitive_ico_sphere_add(radius=0.1, location=(x, y, z))

  # Set the render resolution
  bpy.context.scene.render.resolution_x = CAMERA_IMAGE_SIZE[0]
  bpy.context.scene.render.resolution_y = CAMERA_IMAGE_SIZE[1]

  # Render the image and save it to a file
  bpy.ops.render.render(write_still=True, use_viewport=True)
  bpy.data.images['Render Result'].save_render(
    filepath=f'{IMAGE_FILEPATH}{theta:.4f}_{phi:.4f}.png',
  )