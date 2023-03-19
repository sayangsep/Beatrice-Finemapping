# BEATRICE: Bayesian Fine-mapping from Summary Data using Deep Variational Inference

In this repository, we introduce BEATRICE, a finemapping tool to identify putative causal variants from GWAS summary data. BEATRICE combines a hierarchical Bayesian model with a deep learning-based inference procedure. This combination provides greater inferential power to handle noise and spurious interactions due to polygenicity of the trait, trans-interactions of variants, or varying correlation structure of the genomic region. 

Table of contents:

 - [Installation](#installation)
    - [Install using Singularity](#install-beatrice-using-singularity)
    - [Install using Anaconda](#install-beatrice-using-anaconda)
    - [Install using Python packages](#install-beatrice-using-python-packages)
 - [Usage](#usage)
    - [Using Container](#run-beatrice-using-singularity-container)
    - [Using Anaconda](#run-beatrice-using-anaconda-environment)
    - [Using Python](#run-beatrice-using-anaconda-environment)
 - [Miscellaneous](#miscellaneous)
 
 ## Installation
 
 ### Install BEATRICE Using Singularity
 We have uploaded the singularity container (.sif) to run BEATRICE. It is highly recommended to run BEATRICE inside the container. The container was built using [Singularity](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html). Please follow the [installation steps](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) to install singularity in your system. Once installed, you can pull the .sif file using the following command:
 ```
 singularity pull --arch amd64 library://sayan_ghosal/finemapping/beatrice.sif:latest
 ```
 The container contains all the dependencies to run BEATRICE. Once download, refer to the section [Run BEATRICE Using Singularity Container](#run-beatrice-using-singularity-container) to learn about using BEATRICE.
 
 ### Install BEATRICE Using Anaconda
 We have uploaded the [anaconda](https://docs.anaconda.com/anaconda/install/index.html) environment file, [`conda_environment.yml`](https://github.com/sayangsep/Beatrice-Finemapping/blob/main/conda_environment.yml), listing all the dependencies required to run BEATRICE. The anaconda environment can be easily created by the following command:
 ```
 conda env create -f conda_environment.yml
 ```
 
 This will create an environment named, `beatrice_env`, which will have all the dependencies downloaded inside.  Once installed, refer to the section [Run BEATRICE Using Anaconda Environment](#run-beatrice-using-anaconda-environment) to learn about using BEATRICE using anaconda.
 
 ### Install BEATRICE Using Python packages
 
 The user can also run BEATRICE with their own personal installation of python packages. The packages required for running BEATRICE are listed below:
 
 - [PyTorch](https://pytorch.org/)
 - [Absl](https://anaconda.org/anaconda/absl-py)
 - [Matplotlib](https://matplotlib.org/stable/users/installing/index.html)
 - [Numpy](https://numpy.org/install/)
 - [Glob](https://pypi.org/project/glob2/)
 - [Pickle](https://anaconda.org/conda-forge/pickle5)
 - [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)
 - [Seaborn](https://seaborn.pydata.org/installing.html)
 - [Imageio](https://imageio.readthedocs.io/en/v2.8.0/installation.html)
 - [Shutil](https://anaconda.org/conda-forge/pytest-shutil)
 
 ## Usage
 
 ### Run BEATRICE Using Singularity Container
 
  ### Run BEATRICE Using Anaconda Environment
  
  ### Run BEATRICE Using Python Packages
 
 ## Miscellaneous
