"""
Animation Saving Module for Digital Option Pricing

This module saves the animation of the digital call option pricing simulation as a GIF.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import time
import os
import sys
import subprocess

# Import custom modules
from pricing_models import bs_digital_call_price, monte_carlo_price, calculate_error
from simulation import simulate_paths
from visualization import create_animation

def create_gif_from_frames(frame_files, output_file, duration=100):
    """
    Create a GIF from a list of image files.
    
    Args:
        frame_files (list): List of image file paths
        output_file (str): Output GIF file path
        duration (int): Duration of each frame in milliseconds
    """
    try:
        # Try to install and use Pillow if not already installed
        try:
            from PIL import Image
        except ImportError:
            print("Installing Pillow...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
            from PIL import Image
        
        print("Creating GIF using Pillow...")
        frames = [Image.open(f) for f in frame_files]
        
        # Save GIF
        frames[0].save(
            output_file,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=duration,
            loop=0
        )
        print(f"GIF created successfully: {output_file}")
        return True
    except Exception as e:
        print(f"Error creating GIF: {e}")
        return False

def main():
    """Main function to create and save animation."""
    # Create directory for images if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
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
    
    # Pre-calculate prices and errors for each path count
    mc_prices = []
    errors = []
    times = []  # Keep this for compatibility with visualization function
    
    for count in path_counts:
        start_time = time.time()
        mc_price = monte_carlo_price(all_paths[:count], K, r, T, H)
        end_time = time.time()
        
        mc_prices.append(mc_price)
        errors.append(calculate_error(mc_price, bs_price))
        times.append(end_time - start_time)  # Keep this for compatibility
    
    # Create animation
    ani, fig = create_animation(
        all_paths, path_counts, time_points, K, T, r, H, 
        bs_price, mc_prices, errors, times
    )
    
    # Save each frame as an image
    frame_files = []
    for frame in range(len(path_counts)):
        ani._func(frame)
        frame_file = f'images/frame_{frame:03d}.png'
        plt.savefig(frame_file, dpi=100)
        frame_files.append(frame_file)
        print(f"Saved frame {frame+1}/{len(path_counts)}")
    
    # Create GIF from frames
    gif_file = 'images/simulation_preview.gif'
    if create_gif_from_frames(frame_files, gif_file, duration=1000):
        print(f"Animation saved as {gif_file}")
    else:
        print("Could not create GIF. Individual frames were saved in the 'images' directory.")
    
    print("Animation generation completed.")

if __name__ == "__main__":
    main() 