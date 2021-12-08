# Tanzanian Water Pumps
This repository contains all the code needed for my DS 6040 final project.
The purpose of the project is to develop a Bayesian estimator to predict the probability a well is broken.
This prediction is then fed into an optimization alogorithm to strategically allocate resources.

# Instructions to Run Code
If you are interested in reproducing or extending the work for this project,
please heed the following steps.

## Cloning the GitHub Repo
To run the code you will need to clone the repository.
These instruction assume you already have Git on your machine.
If not, go to https://git-scm.com/downloads and follow their directions.

In your desired file storage location run the following command

```console
git clone https://github.com/zblanks/water_pumps.git
```

## Obtaining a MOSEK Academic License
I determined the optimal waterpoints to repair by formulating and solving a MINLP.
MINLPs are a difficult problem class,
and usually require a commerical solver to scale to larger problems.
For this project,
I used the MOSEK commerical solver with a free academic license.
If you are at an academic institution, 
follow these steps to get the necessary license for this problem:

1. Request a personal academic license at https://www.mosek.com/products/academic-licenses/
2. After you're approved,
download the license file to the recommended location provided in the welcome email

## Building the Conda Environment
I recommend building a separate Conda environment to run the code.
These instructions assume you already downloaded the Anaconda package manager,
and the `conda` command can be executed in your terminal.
If not, go to https://docs.conda.io/en/latest/miniconda.html
to download the latest version of miniconda and follow their instructions.

Using the `environment.yml` file included in the repo,
and in the same file location you cloned the repo,
execute the following command in your terminal

```console
conda env create -f environment.yml
```

This will get all the necessary packages needed to run the code.
Unfortunately I only tested the code on my local machine,
but in the future I intend to check that it works for other operating systems.

## Obtaining the Data
The data I used for this project is hosted by DrivenData.
Please follow these instruction to obtain a working copy:

1. Go to the [website](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/)
2. Create an account and agree to the terms and conditions
3. Go to the "Data Download" tab
4. Download "Training Set Values" and "Training Set Labels"
    * I titled these "X.csv" and "y.csv", respectively

You will also need to correct the data file location to match your machine.
Unfortunately for this project I hard-coded these values,
but I intend to find a more suitable and general storage method
to make this step unnecessary.
In particular, in the `water_pumps/run_experiment.py` file,
locate the lines with the variables `xfile` and `yfile`,
and change their paths to the location on your machine.

## Running the Experiment
After cloning the repo, obtaining a MOSEK license, building the package environment,
and getting the data, you're ready to run the model.
In the same location where you cloned the repo,
execute the following commands

```console
conda activate water_pumps
python water_pumps/run_experiment.py
```

This sequence of commands activates the Conda environment
and runs the set of experiments for this project.
There may be some deprecation warnings with how NumPyro is calling the Jax library,
but everything should run without an errors.

# Future Areas of Work to Improve Reproducibility
This repo contains all of the code necessary to run the experiments for the project;
however, there are a number of items to make this easier to reproduce.

1. Building a Docker container to automate the steps for running the code
2. Hosting the data in a cloud-based location
3. Significantly beefing up the test suite
