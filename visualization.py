"""
Visualization Module for Digital Option Pricing

This module contains functions for visualizing the results of digital option pricing simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import time

def create_animation(all_paths, path_counts, time_points, K, T, r, H, bs_price, mc_prices, errors, times):
    """
    Create an animation of the digital option pricing simulation.
    
    Args:
        all_paths (numpy.ndarray): All simulated stock price paths
        path_counts (list): List of path counts for each frame
        time_points (numpy.ndarray): Time points for the simulation
        K (float): Strike price
        T (float): Time to maturity
        r (float): Risk-free interest rate
        H (float): Payoff amount
        bs_price (float): Black-Scholes price
        mc_prices (list): List of Monte Carlo prices for each path count
        errors (list): List of errors for each path count
        times (list): List of computation times for each path count
        
    Returns:
        FuncAnimation: Animation object
    """
    # Initialize figure
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    
    # Set up subplots
    ax_paths = fig.add_subplot(gs[0, 0])  # Stock price paths
    ax_dist = fig.add_subplot(gs[0, 1])   # Terminal stock price distribution
    ax_conv = fig.add_subplot(gs[1, :])   # Monte Carlo price convergence
    ax_error = fig.add_subplot(gs[2, 0])  # Error vs. path count
    ax_time = fig.add_subplot(gs[2, 1])   # Time complexity
    
    # Set title
    S0 = all_paths[0, 0]  # Initial stock price
    sigma = np.sqrt(2 * ((np.log(all_paths[:, 1] / S0) / np.sqrt(time_points[1])).var()))  # Estimate sigma
    
    fig.suptitle(f"Digital Call Option Pricing Simulation\nS={S0}, K={K}, T={T}, r={r}, σ={sigma:.2f}, H={H}", fontsize=16)
    
    # Reserve space for info text
    plt.subplots_adjust(bottom=0.15)
    
    # Create info text object
    info_text = fig.text(0.5, 0.05, "", ha="center", fontsize=12, 
                        bbox={"facecolor":"white", "alpha":0.8, "pad":5, "boxstyle":"round"})
    
    # Animation initialization function
    def init():
        # 1. Initialize stock price path plot
        ax_paths.clear()
        ax_paths.set_xlim(0, T)
        ax_paths.set_ylim(60, 160)
        ax_paths.set_xlabel('Time (years)')
        ax_paths.set_ylabel('Stock Price')
        ax_paths.set_title('Stock Price Paths')
        ax_paths.axhline(y=K, color='r', linestyle='--', label=f'Strike K={K}')
        ax_paths.legend()
        
        # 2. Initialize terminal stock price distribution
        ax_dist.clear()
        ax_dist.set_xlim(50, 200)
        ax_dist.set_ylim(0, 250)
        ax_dist.set_xlabel('Stock Price')
        ax_dist.set_ylabel('Frequency')
        ax_dist.set_title('Terminal Stock Price Distribution')
        ax_dist.axvline(x=K, color='r', linestyle='--', label=f'Strike K={K}')
        ax_dist.legend()
        
        # 3. Initialize Monte Carlo price convergence
        ax_conv.clear()
        ax_conv.set_xscale('log')
        ax_conv.set_xlim(path_counts[0]/2, path_counts[-1]*2)
        ax_conv.set_ylim(-0.5, 1.5)
        ax_conv.set_xlabel('Number of Paths')
        ax_conv.set_ylabel('Option Price')
        ax_conv.set_title('Monte Carlo Price Convergence')
        ax_conv.axhline(y=bs_price, color='r', linestyle='--', label=f'BS Price = {bs_price:.5f}')
        ax_conv.legend()
        
        # 4. Initialize error vs. path count
        ax_error.clear()
        ax_error.set_xscale('log')
        ax_error.set_yscale('log')
        ax_error.set_xlim(path_counts[0]/2, path_counts[-1]*2)
        ax_error.set_ylim(1e-4, 1)
        ax_error.set_xlabel('Number of Paths')
        ax_error.set_ylabel('Absolute Error')
        ax_error.set_title('Error vs. Path Count')
        
        # 5. Initialize time complexity
        ax_time.clear()
        ax_time.set_xscale('log')
        ax_time.set_yscale('log')
        ax_time.set_xlim(path_counts[0]/2, path_counts[-1]*2)
        ax_time.set_ylim(1e-6, 1e-1)
        ax_time.set_xlabel('Number of Paths')
        ax_time.set_ylabel('Computation Time (s)')
        ax_time.set_title('Time Complexity')
        
        # Plot initial data
        ax_time.plot(path_counts, times, 'b-', marker='o', label='Time')
        
        # O(N^0.36) theoretical time complexity
        theoretical_time = times[0] * (np.array(path_counts) / path_counts[0]) ** 0.36
        ax_time.plot(path_counts, theoretical_time, 'r--', label='Rate: O(N^0.36)')
        
        ax_time.legend()
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.15)
        
        # Initialize info text
        info_text.set_text("Starting simulation...")
        
        return []
    
    # Animation update function
    def update(frame):
        # Get path count for current frame
        path_count = path_counts[frame]
        current_paths = all_paths[:path_count]
        
        # Use pre-calculated values
        mc_price = mc_prices[frame]
        error = errors[frame]
        computation_time = times[frame]
        
        # Calculate terminal stock prices
        terminal_prices = current_paths[:, -1]
        in_money = terminal_prices > K
        
        # 1. Update stock price path plot
        ax_paths.clear()
        ax_paths.set_xlim(0, T)
        ax_paths.set_ylim(60, 160)
        ax_paths.set_xlabel('Time (years)')
        ax_paths.set_ylabel('Stock Price')
        ax_paths.set_title('Stock Price Paths')
        
        # Display at most 100 paths
        visible_paths = min(100, path_count)
        for i in range(visible_paths):
            ax_paths.plot(time_points, current_paths[i], 'b-', alpha=0.3, linewidth=0.5)
        
        ax_paths.axhline(y=K, color='r', linestyle='--', label=f'Strike K={K}')
        ax_paths.legend()
        
        # 2. Update terminal stock price distribution
        ax_dist.clear()
        ax_dist.set_xlim(50, 200)
        ax_dist.set_ylim(0, 250)
        ax_dist.set_xlabel('Stock Price')
        ax_dist.set_ylabel('Frequency')
        ax_dist.set_title('Terminal Stock Price Distribution')
        
        # Out-of-money (red)
        out_of_money_prices = terminal_prices[~in_money]
        if len(out_of_money_prices) > 0:
            ax_dist.hist(out_of_money_prices, bins=20, color='red', 
                       alpha=0.7, label=f'Out-of-money: {len(out_of_money_prices)}')
        
        # In-the-money (green)
        in_money_prices = terminal_prices[in_money]
        if len(in_money_prices) > 0:
            ax_dist.hist(in_money_prices, bins=20, color='green', 
                       alpha=0.7, label=f'In-the-money: {len(in_money_prices)}')
        
        ax_dist.axvline(x=K, color='r', linestyle='--', label=f'Strike K={K}')
        ax_dist.legend()
        
        # 3. Update Monte Carlo price convergence
        ax_conv.clear()
        ax_conv.set_xscale('log')
        ax_conv.set_xlim(path_counts[0]/2, path_counts[-1]*2)
        ax_conv.set_ylim(-0.5, 1.5)
        ax_conv.set_xlabel('Number of Paths')
        ax_conv.set_ylabel('Option Price')
        ax_conv.set_title('Monte Carlo Price Convergence')
        
        # Display only data up to current frame
        current_path_counts = path_counts[:frame+1]
        current_mc_prices = mc_prices[:frame+1]  # Price data up to current frame
        
        ax_conv.plot(current_path_counts, current_mc_prices, 'b-', marker='o', label='MC Price')
        ax_conv.axhline(y=bs_price, color='r', linestyle='--', label=f'BS Price = {bs_price:.5f}')
        
        # Calculate and display 95% confidence interval
        if frame > 0:
            # Calculate confidence interval only for current path counts
            std_error = []
            for i, count in enumerate(current_path_counts):
                # Calculate standard error of payoffs for each path count
                payoffs = np.where(all_paths[:count, -1] > K, H, 0)
                std_error.append(np.std(payoffs) / np.sqrt(count))
            
            std_error = np.array(std_error)
            upper_bound = np.array(current_mc_prices) + 1.96 * std_error
            lower_bound = np.array(current_mc_prices) - 1.96 * std_error
            
            ax_conv.fill_between(current_path_counts, lower_bound, upper_bound, 
                               color='green', alpha=0.2, label='95% Confidence')
        
        ax_conv.legend()
        
        # 4. Update error vs. path count
        ax_error.clear()
        ax_error.set_xscale('log')
        ax_error.set_yscale('log')
        ax_error.set_xlim(path_counts[0]/2, path_counts[-1]*2)
        ax_error.set_ylim(1e-4, 1)
        ax_error.set_xlabel('Number of Paths')
        ax_error.set_ylabel('Absolute Error')
        ax_error.set_title('Error vs. Path Count')
        
        current_errors = errors[:frame+1]  # Error data up to current frame
        ax_error.plot(current_path_counts, current_errors, 'b-', marker='o', label='MC Error')
        
        # O(N^(-0.36)) theoretical convergence rate (observed from image)
        if frame > 0:
            theoretical_rate = current_errors[0] * (np.array(current_path_counts) / current_path_counts[0]) ** (-0.36)
            ax_error.plot(current_path_counts, theoretical_rate, 'r--', label='Rate: O(N^(-0.36))')
        
        ax_error.legend()
        
        # 5. Update time complexity
        ax_time.clear()
        ax_time.set_xscale('log')
        ax_time.set_yscale('log')
        ax_time.set_xlim(path_counts[0]/2, path_counts[-1]*2)
        ax_time.set_ylim(1e-6, 1e-1)
        ax_time.set_xlabel('Number of Paths')
        ax_time.set_ylabel('Computation Time (s)')
        ax_time.set_title('Time Complexity')
        
        current_times = times[:frame+1]  # Time data up to current frame
        ax_time.plot(current_path_counts, current_times, 'b-', marker='o', label='Time')
        
        # O(N^0.36) theoretical time complexity (observed from image)
        if frame > 0:
            theoretical_time = current_times[0] * (np.array(current_path_counts) / current_path_counts[0]) ** 0.36
            ax_time.plot(current_path_counts, theoretical_time, 'r--', label='Rate: O(N^0.36)')
        
        ax_time.legend()
        
        # Update info text
        info_text.set_text(f"Paths: {path_count}, MC Price: {mc_price:.5f}, BS Price: {bs_price:.5f}, Error: {error:.5f}")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.15)
        
        return []
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(path_counts),
                       init_func=init, interval=1000, blit=False, repeat=True)
    
    return ani, fig 