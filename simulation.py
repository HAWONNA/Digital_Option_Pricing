"""
Stock Price Simulation Module

This module provides functions for simulating stock price paths
using Geometric Brownian Motion (GBM).
"""

import numpy as np

def simulate_paths(S0, T, r, sigma, n_paths, n_steps):
    """
    Simulate stock price paths using Geometric Brownian Motion.
    
    Args:
        S0 (float): Initial stock price
        T (float): Time horizon in years
        r (float): Risk-free interest rate
        sigma (float): Volatility
        n_paths (int): Number of paths to simulate
        n_steps (int): Number of time steps
        
    Returns:
        numpy.ndarray: Array of shape (n_paths, n_steps+1) containing simulated paths
    """
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    return paths 