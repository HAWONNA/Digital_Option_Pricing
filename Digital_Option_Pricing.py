import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import time
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
import scipy.stats as stats
import matplotlib
matplotlib.use('TkAgg')  # Set backend to TkAgg

# Parameter settings
S0 = 100      # Initial stock price
K = 105       # Strike price
T = 1.0       # Maturity (years)
r = 0.05      # Risk-free interest rate
sigma = 0.2   # Volatility
H = 1.0       # Digital option payoff

# Black-Scholes digital call option price (direct implementation of normal cumulative distribution function)
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
def simulate_paths(S0, K, T, r, sigma, n_paths, n_steps):
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

# Error calculation function
def calculate_error(mc_price, bs_price):
    return np.abs(mc_price - bs_price)

# Create app
app = dash.Dash(__name__)

# Set layout
app.layout = html.Div([
    html.H1("Digital Call Option Pricing Simulation", style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            html.Label("Number of Paths:"),
            dcc.Slider(
                id='n-paths-slider',
                min=10,
                max=5000,
                step=10,
                value=100,
                marks={10: '10', 100: '100', 1000: '1000', 5000: '5000'},
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.Button('Start Simulation', id='start-button', n_clicks=0),
            html.Button('Reset', id='reset-button', n_clicks=0),
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
    
    html.Div([
        dcc.Graph(id='simulation-graph'),
    ]),
    
    dcc.Store(id='simulation-data'),
    dcc.Interval(
        id='interval-component',
        interval=500,  # in milliseconds (500ms = 0.5s)
        n_intervals=0,
        disabled=True
    ),
    
    html.Div([
        html.P(f"Parameters: S={S0}, K={K}, T={T}, r={r}, σ={sigma}, H={H}"),
        html.P(id='mc-bs-info')
    ])
])

# Callback functions
@app.callback(
    Output('interval-component', 'disabled'),
    Input('start-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    State('interval-component', 'disabled')
)
def toggle_interval(start_clicks, reset_clicks, disabled):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-button' and start_clicks > 0:
        return False
    elif button_id == 'reset-button' and reset_clicks > 0:
        return True
    return disabled

@app.callback(
    Output('simulation-data', 'data'),
    Input('reset-button', 'n_clicks'),
    Input('n-paths-slider', 'value')
)
def initialize_data(reset_clicks, n_paths):
    # Initialize data
    n_steps = 100
    paths = simulate_paths(S0, K, T, r, sigma, n_paths, n_steps)
    
    # Data for tracking Monte Carlo prices over time
    path_counts = np.logspace(0, np.log10(n_paths), 20).astype(int)
    path_counts = np.unique(path_counts)
    
    mc_prices = []
    errors = []
    times = []
    
    for count in path_counts:
        start_time = time.time()
        mc_price = monte_carlo_price(paths[:count], K, r, T, H)
        end_time = time.time()
        
        mc_prices.append(mc_price)
        errors.append(calculate_error(mc_price, bs_price))
        times.append(end_time - start_time)
    
    # Calculate terminal stock price distribution
    terminal_prices = paths[:, -1]
    in_money = terminal_prices > K
    
    return {
        'paths': paths.tolist(),
        'time_points': np.linspace(0, T, n_steps + 1).tolist(),
        'terminal_prices': terminal_prices.tolist(),
        'in_money': in_money.tolist(),
        'path_counts': path_counts.tolist(),
        'mc_prices': mc_prices,
        'errors': errors,
        'times': times,
        'bs_price': bs_price,
        'current_step': 0,
        'n_paths': n_paths,
        'n_steps': n_steps
    }

@app.callback(
    [Output('simulation-graph', 'figure'),
     Output('mc-bs-info', 'children'),
     Output('simulation-data', 'data', allow_duplicate=True)],
    [Input('interval-component', 'n_intervals'),
     Input('simulation-data', 'data')],
    prevent_initial_call=True
)
def update_graph(n_intervals, data):
    if data is None:
        return dash.no_update, dash.no_update, dash.no_update
    
    paths = np.array(data['paths'])
    time_points = data['time_points']
    terminal_prices = np.array(data['terminal_prices'])
    in_money = np.array(data['in_money'])
    path_counts = np.array(data['path_counts'])
    mc_prices = np.array(data['mc_prices'])
    errors = np.array(data['errors'])
    times = np.array(data['times'])
    bs_price = data['bs_price']
    current_step = data['current_step']
    n_paths = data['n_paths']
    n_steps = data['n_steps']
    
    # Determine current number of paths to display
    if current_step < len(path_counts):
        current_paths = path_counts[current_step]
    else:
        current_paths = n_paths
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 1}, {"colspan": 1}],
            [{"colspan": 2}, None],
            [{"colspan": 1}, {"colspan": 1}]
        ],
        subplot_titles=(
            "Stock Price Paths", "Terminal Stock Price Distribution",
            "Monte Carlo Price Convergence",
            "Error vs. Path Count", "Time Complexity"
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. Stock price paths plot (top left)
    visible_paths = min(100, current_paths)  # Display at most 100 paths
    for i in range(visible_paths):
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=paths[i, :],
                mode='lines',
                line=dict(color='blue', width=0.5),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Display strike price
    fig.add_trace(
        go.Scatter(
            x=[0, T],
            y=[K, K],
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            name=f'Strike K={K}'
        ),
        row=1, col=1
    )
    
    # 2. Terminal stock price distribution (top right)
    if current_step > 0:
        # Use only paths up to current step
        current_terminal_prices = terminal_prices[:current_paths]
        current_in_money = in_money[:current_paths]
        
        # Out-of-money (red)
        out_of_money_prices = current_terminal_prices[~current_in_money]
        if len(out_of_money_prices) > 0:
            fig.add_trace(
                go.Histogram(
                    x=out_of_money_prices,
                    nbinsx=20,
                    marker_color='red',
                    name=f'Out-of-money: {len(out_of_money_prices)}'
                ),
                row=1, col=2
            )
        
        # In-the-money (green)
        in_money_prices = current_terminal_prices[current_in_money]
        if len(in_money_prices) > 0:
            fig.add_trace(
                go.Histogram(
                    x=in_money_prices,
                    nbinsx=20,
                    marker_color='green',
                    name=f'In-the-money: {len(in_money_prices)}'
                ),
                row=1, col=2
            )
        
        # Display strike price
        fig.add_trace(
            go.Scatter(
                x=[K, K],
                y=[0, 30],  # Appropriate y-range needed
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                name=f'Strike K={K}'
            ),
            row=1, col=2
        )
    
    # 3. Monte Carlo price convergence (middle)
    if current_step > 0:
        # Use only data up to current step
        current_path_counts = path_counts[:current_step+1]
        current_mc_prices = mc_prices[:current_step+1]
        
        # MC price
        fig.add_trace(
            go.Scatter(
                x=current_path_counts,
                y=current_mc_prices,
                mode='lines+markers',
                name='MC Price',
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        
        # BS price (exact solution)
        fig.add_trace(
            go.Scatter(
                x=[current_path_counts[0], current_path_counts[-1]],
                y=[bs_price, bs_price],
                mode='lines',
                name=f'BS Price = {bs_price:.5f}',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=1
        )
        
        # 95% confidence interval (example)
        std_error = np.array([np.std(np.where(paths[:count, -1] > K, H, 0)) / np.sqrt(count) for count in current_path_counts])
        upper_bound = np.array(current_mc_prices) + 1.96 * std_error
        lower_bound = np.array(current_mc_prices) - 1.96 * std_error
        
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([current_path_counts, current_path_counts[::-1]]),
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill='toself',
                fillcolor='rgba(0,176,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence'
            ),
            row=2, col=1
        )
    
    # 4. Error vs path count (bottom left)
    if current_step > 0:
        fig.add_trace(
            go.Scatter(
                x=current_path_counts,
                y=errors[:current_step+1],
                mode='lines+markers',
                name='MC Error',
                line=dict(color='blue', width=2)
            ),
            row=3, col=1
        )
        
        # O(N^(-0.5)) theoretical convergence rate
        theoretical_rate = errors[0] * np.sqrt(current_path_counts[0]) / np.sqrt(current_path_counts)
        fig.add_trace(
            go.Scatter(
                x=current_path_counts,
                y=theoretical_rate,
                mode='lines',
                name='Rate: O(N^(-0.5))',
                line=dict(color='red', dash='dash')
            ),
            row=3, col=1
        )
    
    # 5. Time complexity (bottom right)
    if current_step > 0:
        fig.add_trace(
            go.Scatter(
                x=current_path_counts,
                y=times[:current_step+1],
                mode='lines+markers',
                name='Time',
                line=dict(color='green', width=2)
            ),
            row=3, col=2
        )
        
        # O(N) theoretical time complexity
        theoretical_time = times[0] * current_path_counts / current_path_counts[0]
        fig.add_trace(
            go.Scatter(
                x=current_path_counts,
                y=theoretical_time,
                mode='lines',
                name='Rate: O(N)',
                line=dict(color='red', dash='dash')
            ),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=900,
        title_text=f"Digital Call Option Pricing Simulation (S={S0}, K={K}, T={T}, r={r}, σ={sigma}, H={H})",
        title_x=0.5,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Set X-axis log scale
    fig.update_xaxes(type="log", row=2, col=1)
    fig.update_xaxes(type="log", row=3, col=1)
    fig.update_xaxes(type="log", row=3, col=2)
    
    # Set Y-axis log scale
    fig.update_yaxes(type="log", row=3, col=1)
    fig.update_yaxes(type="log", row=3, col=2)
    
    # Set axis labels
    fig.update_xaxes(title_text="Time (years)", row=1, col=1)
    fig.update_yaxes(title_text="Stock Price", row=1, col=1)
    
    fig.update_xaxes(title_text="Stock Price", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    fig.update_xaxes(title_text="Number of Paths", row=2, col=1)
    fig.update_yaxes(title_text="Option Price", row=2, col=1)
    
    fig.update_xaxes(title_text="Number of Paths", row=3, col=1)
    fig.update_yaxes(title_text="Absolute Error", row=3, col=1)
    
    fig.update_xaxes(title_text="Number of Paths", row=3, col=2)
    fig.update_yaxes(title_text="Computation Time (s)", row=3, col=2)
    
    # Update info text
    if current_step < len(path_counts):
        current_mc_price = mc_prices[current_step]
        current_error = errors[current_step]
        info_text = f"Paths: {current_paths}, MC Price: {current_mc_price:.5f}, BS Price: {bs_price:.5f}, Error: {current_error:.5f}"
    else:
        final_mc_price = monte_carlo_price(paths, K, r, T, H)
        final_error = calculate_error(final_mc_price, bs_price)
        info_text = f"Paths: {n_paths}, MC Price: {final_mc_price:.5f}, BS Price: {bs_price:.5f}, Error: {final_error:.5f}"
    
    # Update data
    data['current_step'] = min(current_step + 1, len(path_counts))
    
    return fig, info_text, data

if __name__ == '__main__':
    app.run_server(debug=True)
