This repository contains a small code that explores and tests whether it's interesting to learn a perturbed relaxation for a MILP, the idea is detailed later.

Given a MILP formulation (in this repository's case we choose the CFL problem), we tried the following workflow:
* Solve the problem
* Train a NN that predicts a score for each integer variables that acts as a perturbation. This NN is trained using a FY loss between the integer variables' ground truth and the value of these variables in the perturbed relaxation constructed with the model


---
The `configs` folder in which can you place the configs of the models you want to train, then using the script `scripts/launch_all_config.sh` you can start the training of this models in parallel on the LRZ cluster.
The models are then stored in `saved_models`.

You can clear the execution files (`[...].err`, `[...].out` and the scripts used for the batch training of all the configs). 

--- 
The main script is `script_train_model.py`.