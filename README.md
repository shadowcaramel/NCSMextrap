# NCSMextrap
This is the github repo for the paper [paper name will be added]. Neural network tool for the extrapolation problem of NCSM calculations.

Warning: the code is not well-written, well-documented, contains unnecessary stuff and provided "as is".

This repository icludes following files:
  * config_maker.py - script that produces configuration file "config.ini".
  * E_or_R_train.py - script that includes data preprocessing, neural network building procedure, training, saving predictions etc. Multiprocessing is utilized to speed-up calculations.
  * PP_script.py -  postprocessing script that produce all the results.
  * skeleton.py - script that run above parts. This project was intended to use with [Slurm workload manager](https://slurm.schedmd.com/documentation.html). Dividing the "job" into parts with Slurm is utilized to further speed-up.
  * data_sample.xlsx

Program was tested with Python 3.11, TensorFlow 2.14.0 and TensorFlow Addons 0.22.0.
