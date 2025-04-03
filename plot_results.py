import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt

def plot_from_file():
    # Load saved metrics
    data = np.load('training_metrics.npz', allow_pickle=True)
    metrics = {key: data[key] for key in data.files}
    
    # Plotting code (same as before)
    plt.figure(figsize=(15, 10))
    
    # Portfolio Value
    plt.subplot(2, 2, 1)
    plt.plot(metrics['portfolio_values'])
    plt.title("Portfolio Value Over Time")
    
    # Position Sizing
    plt.subplot(2, 2, 2)
    plt.plot(metrics['positions'])
    plt.title("Position History")
    
    # Return Distribution
    plt.subplot(2, 2, 3)
    returns = np.diff(metrics['portfolio_values'])
    plt.hist(returns, bins=50)
    plt.title("Return Distribution")
    
    # Drawdown
    plt.subplot(2, 2, 4)
    peak = np.maximum.accumulate(metrics['portfolio_values'])
    drawdown = (peak - metrics['portfolio_values']) / peak
    plt.plot(drawdown)
    plt.title("Drawdown Analysis")
    
    plt.tight_layout()
    plt.savefig('backtest_results.png')
    print("Saved plot to backtest_results.png")

if __name__ == "__main__":
    plot_from_file()