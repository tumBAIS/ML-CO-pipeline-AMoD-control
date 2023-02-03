# Learning-based Online Algorithms for Autonomous Mobility-on-Demand fleet control

This software learns a dispatching and rebalancing policy for autonomous-mobility on demand systems using a structured learning enriched combinatorial optimization pipeline.

This method is proposed in:
> Link to paper


This repository contains all relevant scripts and data sets to reproduce the results from the paper. We assume using *slurm*.

![](./visualization/movie_numVeh300_09-10.gif)


The structure of the repository is as follows:
- `cplusplus`: contains a C++ interface to run computationally intensive functions in C++
- `data`: contains all relevant data to reproduce the data
- `pipeline`: contains code which specifies the objects of the pipeline
- `prep`: contains code to preprocess the data
- `src`: contains helper files
- `visualization`: contains scripts to reproduce plots and gif
- `create_full_information_solution.py`: Script to solve a full-information problem
- `create_training_instances.py`: Script to rebuild a digraph solution from the full-information solution
- `evaluation.py`: Script to test benchmarks
- `master_script.sh`: Script to start creation of training data / training / evaluation
- `run_bash_benchmarks.cmd`: Script to evaluate benchmarks
- `run_bash_createInstances.cmd`: Script to rebuild digraph solution from full-information solution
- `run_bash_training.cmd`: Script to start training.py to train the *policy_SB* and *policy_CB*
- `run_pipeline.py`: Main file to process the pipeline
- `training.py`: Script to train the *policy_SB* and *policy_CB*



## Overview
The execution of the code follows a three-step approach:
1. **Creation of training instances**: Calculation of full-information solution and rebuilding the digraph solution for online instances
2. **Training**: Minimization of structured learning loss
3. **Evaluation of learned policies**: *policy_CB* and *policy_SB* as well as benchmark policies *sampling*, *greedy*, and *full-information*.


## Creation of training instances
Specify the experiment in the *RUNNING_TYPE* variable in `master_script.sh`.  
To create training instances (full-information solution + generation of training instances), enable the line which calls the `run_bash_createInstances.cmd` script.  
All pre-specified parameters match the parameters used in the paper.  

## Training
Keep the same experiment in the *RUNNING_TYPE* variable in `master_script.sh` as for calculating the training instances.  
In the `master_script.sh` file, enable the line which calls the `run_bash_training.cmd` script.  
This script automatically trains the *policy_SB* and the *policy_CB* for the specified experiment.  
Then, run the `master_script.sh` file.

## Evaluation
The `run_bash_training.cmd` script automatically evaluates the performance of the training right after the training terminated.  
In the `run_bash_training.cmd` script you can define to evaluate the performance on a validation data set or on a testing data set. Both data sets are disjunct from the data set used for training.  

### Benchmarks
Keep the same experiment in the *RUNNING_TYPE* variable in `master_script.sh` as for training the *policy_SB* and the *policy_CB*.  
In the `master_script.sh` file enable the line which runs `run_bash_benchmarks.cmd`.  
In the `run_bash_benchmarks.cmd` script you can define to evaluate the performance on a validation data set or on a testing data set. Both data sets are disjunct from the data set used for training.  
Then, run the `master_script.sh` file.  
The script automatically evaluates the *offline*, *sampling*, and *offline* benchmark.  


## Visualization
- `visualization/visualization_results.py:` generates the plots from the paper
- `visualization/visualization_gif.py`: generates the gif presented on the top of this page
- `visualization/visualization_heatmap.py:` generates the heatmap from the paper showcasing the vehicle distribution for the different benchmarks (Requirement is to first run `visualization/visualization_gif`).