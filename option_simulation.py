import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend to TkAgg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import time

# Parameter settings
S0 = 100      # Initial stock price
K = 105       # Strike price
T = 1.0       # Maturity (years)
r = 0.05      # Risk-free interest rate
sigma = 0.2   # Volatility
H = 1.0       # Digital option payoff

# Black-Scholes digital call option price
def norm_cdf(x):
    return (1.0 + np.math.erf(x / np.sqrt(2.0))) / 2.0

def bs_digital_call_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * norm_cdf(d2)

# Calculate exact BS price
bs_price = bs_digital_call_price(S0, K, T, r, sigma)
print(f"Black-Scholes Digital Call Option Price: {bs_price:.5f}")

# Stock price path simulation function
def simulate_paths(S0, T, r, sigma, n_paths, n_steps):
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    return paths

# Monte Carlo price calculation function
def monte_carlo_price(paths, K, r, T, H):
    terminal_prices = paths[:, -1]
    payoffs = np.where(terminal_prices > K, H, 0)
    return np.exp(-r * T) * np.mean(payoffs)

# Simulation settings
n_paths_max = 5000
n_steps = 100
np.random.seed(42)  # Set seed for reproducibility

# Generate all paths in advance
all_paths = simulate_paths(S0, T, r, sigma, n_paths_max, n_steps)
time_points = np.linspace(0, T, n_steps + 1)

# Path count sequence
path_counts = [10, 20, 45, 91, 200, 500, 1000, 2000, 3531]

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
fig.suptitle(f"Digital Call Option Pricing Simulation\nS={S0}, K={K}, T={T}, r={r}, σ={sigma}, H={H}", fontsize=16)

# Lists for storing data
mc_prices = []
errors = []
times = []

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
    ax_error.legend()
    
    # 5. Initialize time complexity
    ax_time.clear()
    ax_time.set_xscale('log')
    ax_time.set_yscale('log')
    ax_time.set_xlim(path_counts[0]/2, path_counts[-1]*2)
    ax_time.set_ylim(1e-6, 1e-1)
    ax_time.set_xlabel('Number of Paths')
    ax_time.set_ylabel('Computation Time (s)')
    ax_time.set_title('Time Complexity')
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
    
    # Calculate for current path count
    start_time = time.time()
    mc_price = monte_carlo_price(current_paths, K, r, T, H)
    end_time = time.time()
    
    mc_prices.append(mc_price)
    errors.append(abs(mc_price - bs_price))
    times.append(end_time - start_time)
    
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
    current_mc_prices = mc_prices.copy()  # Price data up to current frame
    
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
    
    current_errors = errors.copy()  # Error data up to current frame
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
    
    current_times = times.copy()  # Time data up to current frame
    ax_time.plot(current_path_counts, current_times, 'g-', marker='o', label='Time')
    
    # O(N^0.36) theoretical time complexity (observed from image)
    if frame > 0:
        theoretical_time = current_times[0] * (np.array(current_path_counts) / current_path_counts[0]) ** 0.36
        ax_time.plot(current_path_counts, theoretical_time, 'r--', label='Rate: O(N^0.36)')
    
    ax_time.legend()
    
    # Update info text
    info_text.set_text(f"Paths: {path_count}, MC Price: {mc_price:.5f}, BS Price: {bs_price:.5f}, Error: {errors[-1]:.5f}")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    
    return []

# Create animation
ani = FuncAnimation(fig, update, frames=len(path_counts),
                   init_func=init, interval=1000, blit=False, repeat=True)

# Display animation
plt.show(block=True)

print("Animation completed.") 