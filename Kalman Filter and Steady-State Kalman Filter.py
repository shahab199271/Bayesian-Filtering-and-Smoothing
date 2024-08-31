
#Kalman Filter and Steady-State Kalman Filter
#shahab baloochi


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

np.random.seed(31415926)

# Utility functions

def mv_normal(m, C):
    """ Samples a multivariate normal of mean m and covariance C """
    sample = np.random.randn(*m.shape)
    return m + np.linalg.cholesky(C) @ sample

def rmse(x, y):
    """ Root mean square error between two vectors x and y """
    return np.sqrt(np.mean(np.square(x - y)))

def filter_routine(initial_guess, initial_guess_covariance, update_method, observations):
    """ Loops over the observations and calls the implemented update_method """
    T = observations.shape[0]
    x = initial_guess.copy()
    cov = initial_guess_covariance.copy()
    
    states = np.empty((T, *x.shape))
    covariances = np.empty((T, *initial_guess_covariance.shape))
    
    for i, y in enumerate(observations):
        x, cov = update_method(x, cov, y)
        states[i, :] = x
        covariances[i, :] = cov
    
    return states, covariances

def plot_results(filtered_states, true_states, observations, title, filtered_covariances=None):
    fig, axes = plt.subplots(nrows=2, figsize=(12, 10), sharex=True)
    
    T = observations.shape[0]
    
    axes[0].plot(filtered_states[:, 0], label="Estimated signal", color="b")
    axes[0].plot(true_states[:, 0], label="True signal", linestyle='--', color="r")
    axes[0].scatter(np.arange(T), observations, marker="o", color="g", label="Measurements")    
    
    axes[1].plot(filtered_states[:, 1], label="Estimated derivative", color="b")
    axes[1].plot(true_states[:, 1], label="True derivative", linestyle='--', color="r")
    
    error = rmse(filtered_states, true_states)
    
    for ax in axes.flatten():
        ax.legend(loc="upper left")
        ax.grid(True)
    
    if filtered_covariances is not None:
        axes[0].fill_between(np.arange(T), 
                             filtered_states[:, 0] - np.sqrt(filtered_covariances[:, 0, 0]),
                             filtered_states[:, 0] + np.sqrt(filtered_covariances[:, 0, 0]),
                             alpha=0.33, 
                             color="b")
        axes[1].fill_between(np.arange(T), 
                             filtered_states[:, 1] - np.sqrt(filtered_covariances[:, 1, 1]),
                             filtered_states[:, 1] + np.sqrt(filtered_covariances[:, 1, 1]),
                             alpha=0.33, 
                             color="b")
        fig.suptitle(f"{title} and confidence intervals - RMSE: {error:.3f}")
    else:
        fig.suptitle(f"{title} - RMSE = {error:.3f}")

    plt.show()

# Get data

w = 0.5
q = 0.01
r = 0.1

T = 100
x_0 = np.array([0, 0.1])

def get_data(w, q, r, x_0, T):
    """ Generates data according to the state space model above """
    Q = 0.5 * q * np.array([[(w - np.cos(w) * np.sin(w)) / w ** 3, np.sin(w) ** 2 / w ** 2],
                            [np.sin(w) ** 2 / w ** 2, (w + np.cos(w) * np.sin(w)) / w]])
    
    A = np.array([[np.cos(w), np.sin(w) / w], 
                  [-w * np.sin(w), np.cos(w)]])
    
    C = np.array([1, 0])
    
    observations = np.empty(T)
    true_states = np.empty((T, 2))
    
    x = x_0.copy()
    true_states[0] = x
    
    for i in range(T):
        observations[i] = C.dot(x) + np.sqrt(r) * np.random.randn()
        x = mv_normal(A @ x, Q)
        if i < T - 1:
            true_states[i + 1, :] = x
        
    return observations, true_states, Q, A, C

observations, true_states, Q, A, C = get_data(w, q, r, x_0, T)

def plot_state(observations, true_states, T):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(true_states[:, 0], linestyle='--', label="True Signal")
    ax.scatter(np.arange(T), observations, marker='o', label="Measurements")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Signal value")
    ax.legend(loc="upper left")
    ax.grid(True)
    fig.suptitle("Simulated data")
    plt.show()

plot_state(observations, true_states, T)

# Initial Guess
initial_state = np.array([0, 1])
initial_covariance = np.eye(2)

# Baseline solution

def baseline_update_method(x, cov, obs):
    """ A baseline method that registers the measurement as the first component of the state, 
    and the time difference as the second component: this is a rough approximation of the time derivative.
    It does not change the covariance
    """
    return np.array([obs, obs - x[0]]), cov

baseline_states, _ = filter_routine(initial_state, initial_covariance, baseline_update_method, observations)

plot_results(baseline_states, true_states, observations, "Baseline solution", None)

# Kalman Filter

def kf_update(x, cov, obs):
    """ Implements the Kalman equations from Bayesian Filtering and Smoothing, Ch.4.3
    Uses the global variables A, Q, C, r
    """
    # Prediction step
    x_pred = A @ x
    cov_pred = A @ cov @ A.T + Q
    
    # Update step
    S = C @ cov_pred @ C.T + r
    K = cov_pred @ C.T / S
    y_pred = C @ x_pred
    x_new = x_pred + K * (obs - y_pred)
    cov_new = cov_pred - np.outer(K, C) @ cov_pred
    
    return x_new, cov_new

kalman_states, kalman_covariances = filter_routine(initial_state, initial_covariance, kf_update, observations)

plot_results(kalman_states, true_states, observations, "Kalman Filter", kalman_covariances)

# Steady State Kalman Filter

def compute_steady_state_kalman_gain(A, Q, C, r):
    """ Computes the steady state Kalman gain """
    P = Q.copy()
    for _ in range(1000):  # Iterative calculation to reach steady state
        P_new = A @ P @ A.T + Q - (A @ P @ C[:, None] @ C @ P @ A.T) / (C @ P @ C.T + r)
        if np.allclose(P_new, P, atol=1e-8):
            break
        P = P_new
    S = C @ P @ C.T + r
    K = P @ C.T / S
    return K, P

def steady_kf_update(x, cov, obs):
    """ Implements the Kalman equations with steady state assumption, Bayesian Filtering and Smoothing, Exercise 4.5
    Uses the global variables A, Q, C, r
    """
    if not hasattr(steady_kf_update, "K_steady"):
        steady_kf_update.K_steady, steady_kf_update.P_steady = compute_steady_state_kalman_gain(A, Q, C, r)
    
    K = steady_kf_update.K_steady
    P = steady_kf_update.P_steady
    
    # Prediction step
    x_pred = A @ x
    
    # Update step
    y_pred = C @ x_pred
    x_new = x_pred + K * (obs - y_pred)
    
    return x_new, P

steady_kalman_states, steady_kalman_covariances = filter_routine(initial_state, initial_covariance, steady_kf_update, observations)

plot_results(steady_kalman_states, true_states, observations, "Steady Kalman Filter", steady_kalman_covariances)
