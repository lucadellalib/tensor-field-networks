# Tensor Field Network for Rotation Equivariance in 3D Point Cloud Classification

TensorFlow implementation of Tensor Field Networks (https://arxiv.org/abs/1802.08219). Extended version of the code in https://github.com/tensorfieldnetworks/tensorfieldnetworks/tree/949e64ac6e069c2f1bfbcbf30d13f696a970488a. **Batch learning is now supported**. The proposed models are tested on ModelNet40 point cloud dataset (https://modelnet.cs.princeton.edu/). Developed and tested on Ubuntu 18.04 LTS.

---------------------------------------------------------------------------------------------------------

## Requirements

*   Anaconda Python >= 3.6.4 (see https://www.anaconda.com/distribution/);

*   pip (`sudo apt install python3-pip` to install it on Ubuntu 18.04 LTS);

*   virtualenv >= 16.6.0 (`python3 -m pip install --user virtualenv` to install it on Ubuntu 18.04 LTS).

---------------------------------------------------------------------------------------------------------

## Installation

### Create a virtual environment

Clone or download the repository and type the following commands in the root folder:

```python3 -m venv env```

```source env/bin/activate```

Now the virtual environment *env* is active (type `deactivate` if you want to deactivate it).

---------------------------------------------------------------------------------------------------------

### Install the dependencies

To install the dependencies, type the following command in the virtual environment:

```pip install -r requirements.txt```

---------------------------------------------------------------------------------------------------------

### Download the dataset

Read *modelnet/data/README.md* for instructions on how to download ModelNet40 dataset.

---------------------------------------------------------------------------------------------------------

## Usage

*   `python3 train.py` to train the selected model. `--help` option to show the help;

*   `python3 evaluate.py` to evaluate the selected model. `--help` option to show the help;

*   read *modelnet/tools/README.md* for instructions on how to visualize the point clouds.

---------------------------------------------------------------------------------------------------------

## Contact

luca310795@gmail.com

---------------------------------------------------------------------------------------------------------
