# DDPM (WIP)

## What this is about

Just a simple implementation of Denoising Diffusion Probabilistic Model

## What has been done

1. Nothing

## What else needs to be done

1. Set up the Architecture
1. Set up loss function
1. Set up the dataset and dataloader
1. Set up the training, which could be better implemented admittedly.
1. Set up validation to get validation loss.
1. Results visualisation

## How to run

Create the environment (using conda here)

```
conda create --name ______ python=3.8
conda activate _____
pip install -r ./requirements.txt
```

Make sure you change the directory of your data. I used the FashionMNIST and Flowers102.

```
python -m src.main
```

## Visualisation

Go to ./notebooks/model_out.ipynb to see how the model turned out.

## Resources

### From scratch implementations

### Actual Paper

### Helpful Understanding of the Calculations and motivation

### Loss Function
