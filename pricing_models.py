"""
Pricing Models for Digital Call Options

This module contains implementations of:
1. Black-Scholes model for digital call options
2. Monte Carlo simulation for digital call options
"""

import numpy as np

def norm_cdf(x):
    """
    Standard normal cumulative distribution function.
    
    Args:
        x (float): Input value
        
    Returns:
        float: CDF value
    """
    return (1.0 + np.math.erf(x / np.sqrt(2.0))) / 2.0

def bs_digital_call_price(S, K, T, r, sigma):
    """
    Black-Scholes price for a digital call option.
    
    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity in years
        r (float): Risk-free interest rate
        sigma (float): Volatility
        
    Returns:
        float: Option price
    """
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * norm_cdf(d2)

def monte_carlo_price(paths, K, r, T, H):
    """
    Monte Carlo price for a digital call option.
    
    Args:
        paths (numpy.ndarray): Simulated stock price paths
        K (float): Strike price
        r (float): Risk-free interest rate
        T (float): Time to maturity in years
        H (float): Payoff amount
        
    Returns:
        float: Option price
    """
    terminal_prices = paths[:, -1]
    payoffs = np.where(terminal_prices > K, H, 0)
    return np.exp(-r * T) * np.mean(payoffs)

def calculate_error(mc_price, bs_price):
    """
    Calculate absolute error between Monte Carlo and Black-Scholes prices.
    
    Args:
        mc_price (float): Monte Carlo price
        bs_price (float): Black-Scholes price
        
    Returns:
        float: Absolute error
    """
    return np.abs(mc_price - bs_price) 