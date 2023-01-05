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

## Architechture

The current state of the art implementations of NeRF are:
- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- [instant-ngp](https://github.com/NVlabs/instant-ngp)


