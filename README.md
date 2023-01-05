# NeRF-ito

A tiny minimal implementation of NeRF, you can find the code stream [here](https://youtu.be/fbpNR-qnOrI).


## Dependencies

What framework should we use for this project?

- PyTorch
- TensorFlow
- JAX

According to the [Papers with Code](https://paperswithcode.com/trends), PyTorch is the most popular framework for NeRF, so we will use PyTorch.

We install PyTorch with the following command:

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

Before that let's create a virtual environment:

```
pyenv virtualenv nerfito
pyenv activate nerfito
```

Lets test our installation of pytorch

```
python3
>>> import torch
>>> torch.cuda.is_available()
```

Save the dependencies in a file:

```
pip3 freeze --local > requirements.txt
```

## Architechture

The current state of the art implementations of NeRF are:
- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- [instant-ngp](https://github.com/NVlabs/instant-ngp)

The main components for our NeRF implementation are:

- Volume of space that is voxelized
- Camera position and orientation
- Calculating the ray for each pixel
- Calculating the color and opacity at a point on each ray
- Neural network to predict the color and opacity
- Rendering the image from a camera position
- Loss function for rendered image and ground truth image
- Dataset loader for training and testing
- Dataset of images and camera positions
- Training/Testing loops
- Evaluation Code

Neural Network
- Inputs: Point within the volume, and the view direction of that point (in world space)
    - Point within volume: (x, y, z)
    - View direction: (theta, phi)
- Outputs: color, opacity
    - Color: RGB color of the pixel
    - Opacity: how much light is absorbed by the object 
- Architecture: 3 layers of 256 neurons each


model.py
- model class
- forward function
- loss function

utils.py
- ray sampling
- volume sampling
- volume rendering

train.py
- train loop
- test loop

eval.py
- generate one image from a camera position