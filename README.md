# Bayesian-Filtering-and-Smoothing

Kalman Filter and Steady-State Kalman Filter Implementation
This repository contains a Python implementation of the Kalman Filter and Steady-State Kalman Filter for time-series state estimation. The implementation is based on Exercise 4.6 from the book "Bayesian Filtering and Smoothing" by Simo Särkkä.

Table of Contents
Overview
Features
Installation
Usage
Code Structure
Example
Contributing
License
Overview
This repository provides an implementation of:

Kalman Filter (kf_update): A recursive algorithm used to estimate the state of a linear dynamic system from a series of noisy measurements.
Steady-State Kalman Filter (steady_kf_update): A variant of the Kalman Filter that assumes the system has reached a steady state, allowing for a constant Kalman gain to be used in all iterations.
The code generates simulated data according to a state-space model, applies both filtering techniques, and visualizes the results.

Features
Kalman Filter: Dynamic calculation of the Kalman gain at each time step.
Steady-State Kalman Filter: Computation of a steady-state Kalman gain for constant-time filtering.
Visualization: Plotting functions to compare the estimated states with the true states and observations.
Utility Functions: Includes functions for generating simulated data, calculating RMSE, and more.
Installation
Prerequisites
Python 3.x
numpy
matplotlib
scipy
Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/kalman-filter-implementation.git
cd kalman-filter-implementation
Install Dependencies
You can install the required Python packages using pip:

bash
Copy code
pip install numpy matplotlib scipy
Usage
You can run the script directly to generate data, apply the Kalman Filter, and visualize the results:

bash
Copy code
python kalman_filter.py
Running in a Jupyter Notebook
If you prefer to run the code in a Jupyter Notebook, you can open kalman_filter.ipynb and execute the cells interactively.

Code Structure
kalman_filter.py: Main script containing the implementation of the Kalman Filter and Steady-State Kalman Filter, along with the data generation and plotting functions.
README.md: This document, providing an overview and instructions for using the code.
Example
The following is a quick overview of how the filters are applied:

Generate Simulated Data:

A time series of states and noisy observations is generated according to a defined state-space model.
Apply Kalman Filter:

The kf_update function is used to estimate the state at each time step.
Apply Steady-State Kalman Filter:

The steady_kf_update function is used to estimate the state with a constant Kalman gain.
Plot Results:

The estimated states are plotted against the true states and observations.

Contributing
Contributions are welcome! If you have any improvements or bug fixes, feel free to open an issue or submit a pull request.

How to Contribute
Fork the repository.
Create a new branch: git checkout -b feature-name.
Make your changes and commit them: git commit -m 'Add some feature'.
Push to the branch: git push origin feature-name.
Open a pull request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
