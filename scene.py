"""Scene manipulation."""

import bpy

# Get the object called "Camera"
obj = bpy.data.objects["Camera"]

# Print the 3x3 rotation matrix of the object
print(obj.matrix_world.to_3x3())