"""
Digital Call Option Pricing Simulation - Main Module

This is the main module that runs the digital call option pricing simulation.
It integrates the pricing models, simulation, and visualization components.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib
matplotlib.use('TkAgg')  # Set backend to TkAgg

# Import custom modules
from pricing_models import bs_digital_call_price, monte_carlo_price, calculate_error
from simulation import simulate_paths
from visualization import create_animation

def main():
    """Main function to run the simulation."""
    # Parameter settings
    S0 = 100      # Initial stock price
    K = 105       # Strike price
    T = 1.0       # Maturity (years)
    r = 0.05      # Risk-free interest rate
    sigma = 0.2   # Volatility
    H = 1.0       # Digital option payoff
    
    # Calculate Black-Scholes price
    bs_price = bs_digital_call_price(S0, K, T, r, sigma)
    print(f"Black-Scholes Digital Call Option Price: {bs_price:.5f}")
    
    # Simulation settings
    n_paths_max = 5000
    n_steps = 100
    np.random.seed(42)  # Set seed for reproducibility
    
    # Generate all paths in advance
    all_paths = simulate_paths(S0, T, r, sigma, n_paths_max, n_steps)
    time_points = np.linspace(0, T, n_steps + 1)
    
    # Path count sequence
    path_counts = [10, 20, 45, 91, 200, 500, 1000, 2000, 3531]
    
    # Pre-calculate prices, errors, and times for each path count
    mc_prices = []
    errors = []
    times = []
    
    for count in path_counts:
        start_time = time.time()
        mc_price = monte_carlo_price(all_paths[:count], K, r, T, H)
        end_time = time.time()
        
        mc_prices.append(mc_price)
        errors.append(calculate_error(mc_price, bs_price))
        times.append(end_time - start_time)
    
    # Create and display animation
    ani, fig = create_animation(
        all_paths, path_counts, time_points, K, T, r, H, 
        bs_price, mc_prices, errors, times
    )
    
    # Display animation
    plt.show(block=True)
    
    print("Animation completed.")

if __name__ == "__main__":
    main() 