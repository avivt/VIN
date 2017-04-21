# Value Iteration Networks
Code for NIPS 2016 paper:

Value Iteration Networks

Aviv Tamar, Yi Wu, Garrett Thomas, Sergey Levine, and Pieter Abbeel

UC Berkeley


Requires:
- Python (2.7)
- Theano (0.8)

For generating the gridworld data and visualizing results, also requires:
- Matlab (2015 or later required for calling python objects for visualizing trajectories)
- Matlab BGL: http://www.mathworks.com/matlabcentral/fileexchange/10922-matlabbgl
  Put it in matlab_bgl folder.

To start: the scripts directory contains scripts for generating the data, 
and training the different models. 

scripts/make_data_gridworld_nips.m generates the training data (random grid worlds).
Alternatively, you can use the existing data files in the data folder (instead of generating them).

scripts/nips_gridworld_experiments_VIN.sh shows how to train the VIN models.

After training, a weights file (e.g., /results/grid28_VIN.pk) will be created. You can then run:
- script_viz_policy.m to run the trained VIN with the learned weights and view the trajectories
  it produces (line 17 selects the weights file).
- test_network.m to numerically evaluate the learned network on a test set (needs to be generated).


# Related code:
Kent Sommer's implementation of VINs (including data generation) in python + pytorch
https://github.com/kentsommer/pytorch-value-iteration-networks

Abhishek Kumar's implementation of VINs in Tensor Flow
https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks
