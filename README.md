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
 - [Outputs of BEATRICE](#description-of-the-output-files)
 
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
 
 This will create an environment named, `beatrice_env`, which will have all the dependencies downloaded inside.  Once installed, refer to the section [Run BEATRICE Using Anaconda Environment](#run-beatrice-using-anaconda-environment) to learn about using BEATRICE using anaconda. However, please note that this environment file is created with Linux, so it might not work with other operating systems.
 
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
 Download the github repo (https://github.com/sayangsep/Beatrice-Finemapping.git) to your local machine and goto that folder. 
 ### Run BEATRICE Using Singularity Container
 
Running BEATRICE reuqire three things, a 'space' sepereated file storing the z-scores, a 'space' seperated file storing the LD matrix and number of subjects used to run the GWAS. As an example we have proved a file containing [the zcores](example_data/Simulation_data0.z) and [the LD matrix](example_data/Simulation_data0.ld). The zscores file should contain two columns, the first columns is the name of the variants and the second column is the z-scores. 
 
 Once the files are in the above-mentioned format you can run BEATRIC using singularity by taking the following steps:
  - Goto the folder where the github repo is downloaded.
 - Dowload and [the singularity container]((#install-beatrice-using-singularity)) and move it inside the Beatrice-Finemapping folder.
 - Run BEATRICE as, 
 ```
 singularity run beatrice.sif python beatrice.py --z example_data/Simulation_data0.z --LD example_data/Simulation_data0.ld --N 5000 --target results
 ```
 While running on a different data replace the ```--z``` flag with ```--z {location to the z-file}```, and  the ```--LD``` flag with ```--LD {location to the LD file}```, the ```--N``` flag with ```--N {number of subjects}```, and the ```--target``` flag with ```--target {location to store results}```.
 
### Run BEATRICE Using Anaconda Environment
Running BEATRICE require three things, a 'space' sepereated file storing the z-scores, a 'space' seperated file storing the LD matrix and number of subjects used to run the GWAS. As an example we have proved a file containing [the zcores](example_data/Simulation_data0.z) and [the LD matrix](example_data/Simulation_data0.ld). The z-scores file should contain two columns, the first columns is the name of the variants and the second column is the z-scores. 
 
 Once the files are in the above-mentioned format you can run BEATRIC using singularity by taking the following steps:
 - Goto the folder where the github repo is downloaded.
 
 - Install [the anaconda environment](#install-beatrice-using-anaconda) and start it as,
 ```
 conda activate beatrice_env
 ```
 - Run BEATRICE as, 
 ```
 python beatrice.py --z example_data/Simulation_data0.z --LD example_data/Simulation_data0.ld --N 5000
 ```
 While running on a different data replace the ```--z``` flag with ```--z {location to the z-file}```, and  the ```--LD``` flag with ```--LD {location to the LD file}```, the ```--N``` flag with ```--N {number of subjects}```, and the ```--target``` flag with ```--target {location to store results}```.
  
  ### Run BEATRICE Using Python Packages
 Running BEATRICE require three things, a 'space' sepereated file storing the z-scores, a 'space' seperated file storing the LD matrix and number of subjects used to run the GWAS. As an example we have proved a file containing [the zcores](example_data/Simulation_data0.z) and [the LD matrix](example_data/Simulation_data0.ld). The zscores file should contain two columns, the first columns is the name of the variants and the second column is the z-scores. 
 
 Once the files are in the above-mentioned format you can run BEATRIC using singularity by taking the following steps:
 - Goto the folder where the github repo is downloaded.
 
 - Run BEATRICE as, 
 ```
 python beatrice.py --z example_data/Simulation_data0.z --LD example_data/Simulation_data0.ld --N 5000
 ```
 While running on a different data replace the ```--z``` flag with ```--z {location to the z-file}```, and  the ```--LD``` flag with ```--LD {location to the LD file}```, the ```--N``` flag with ```--N {number of subjects}```, and the ```--target``` flag with ```--target {location to store results}```.
 
 ## Description of the output files
