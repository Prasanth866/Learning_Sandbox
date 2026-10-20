from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

def plot_posterior(posterior_dist: dict[str, float], title: str='Posterior Distribution', ax: plt.Axes | None=None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    states = list(posterior_dist.keys())
    probs = list(posterior_dist.values())
    ax.bar(states, probs, color='#4A90E2', alpha=0.8, edgecolor='black')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Probability')
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    return ax

def plot_convergence(history: np.ndarray, exact_value: float, method_name: str, ax: plt.Axes | None=None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    samples = np.arange(1, len(history) + 1)
    ax.plot(samples, history, label=f'{method_name} estimate', color='#E67E22', alpha=0.7)
    ax.axhline(exact_value, color='black', linestyle='--', label='Exact Value')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Probability Estimate')
    ax.set_title(f'Convergence: {method_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
