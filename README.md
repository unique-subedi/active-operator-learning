# active-operator-learning

This repository provides the code for the experiments presented in the paper "[On the Benefits of Active Data Collection in Operator Learning](https://arxiv.org/abs/2410.19725)." 

For user convenience, each Python file in this repository is written to be self-contained. For instance, to implement the estimator alongside the active data collection strategy introduced in the paper for learning the solution operator of heat equations, simply download the file **"linear-estimator-active.py"** from the *heat-equations* directory and execute the code. With basic Python packages installed, this file will:  

1. Generate data using the active collection strategy.  
2. Run the PDE solver to obtain solutions.  
3. Train the estimator.  
4. Produce a plot showing how the mean squared error (computed on 100 test samples) decays as the number of training samples increases.


If you notice any bugs in the code, please feel free to report them at subedi@umich.edu. 

