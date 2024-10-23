# Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning
Thanks for checking out the code for Non-Parametric Transformers (NPTs) code by [Paper](https://arxiv.org/abs/2106.02584).

This codebase will allow you to reproduce experiments from the paper as well as use NPTs for your own research.

## Abstract

We used NPT (Non parametric Transformer Models) for Physics Simulation data (Ordinary Differential Equations), using Physiscs informed Neural Networks and NPT model MSE loss. The experiments show how we can predict the physical system with high accracy, using Transformers. The choice of Gradient Descent Optimization (Adam, Batch-GD, Mini_GD, energy natural gradients, Implicit stochastic gradient descent) also affects the model robustness and capability for prediction. 

## Installation

Set up and activate the Python environment by executing

```
conda env create -f environment.yml
conda activate npt
```

For now, we recommend installing CUDA <= 10.2:

See [issue with CUDA >= 11.0 here](https://github.com/pytorch/pytorch/issues/47908).
 
If you are running this on a system without a GPU, use the above with `environment_no_gpu.yml` instead.

## Examples

We now give some basic examples of running NPT.

NPT downloads all supported datasets automatically, so you don't need to worry about that.

We use [wandb](http://wandb.com/) to log experimental results.
Wandb allows us to conveniently track run progress online.
If you do not want wandb enabled, you can run `wandb off` in the shell where you execute NPT.

For example, run this to explore NPT with default configuration on ODE dataset

```
python run.py --data_set ode
```

You can find all possible config arguments and descriptions in `NPT/configs.py` or using `python run.py --help`.

In `scripts/` we provide a list with the runs and correct hyperparameter configurations presented in the paper.

We hope you enjoy using the code and please feel free to reach out with any questions 😊

