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
 We have uploaded the singularity container (.sif) to run BEATRICE. It is highly recommended to run BEATRICE inside the container. The container was built using [Singularity](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html). Please follow the [installation steps](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) to install singularity in your system. Once installed, you can pull the .sif file using the following commad:
 ```
 singularity pull --arch amd64 library://sayan_ghosal/finemapping/beatrice.sif:latest
 ```
 The container contains all the dependencies to run BEATRICE. Once download, goto [the usage]([#run-beatrice-using-singularity-container]) section to run BEATRICE.
 
 ### Install BEATRICE Using Anaconda
 
 ### Install BEATRICE Using Python packages
 
 ## Usage
 
 ### Run BEATRICE Using Singularity Container
 
  ### Run BEATRICE Using Anaconda Environment
  
  ### Run BEATRICE Using Python Packages
 
 ## Miscellaneous
