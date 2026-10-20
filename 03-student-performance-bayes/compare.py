from __future__ import annotations
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from exact import enumeration_ask, variable_elimination
from sampling import gibbs_sampling, likelihood_weighting, rejection_sampling
from synthetic import chain_network, student_network
OUTPUT_DIR = Path(__file__).resolve().parent / 'figures'

def compute_tvd(p_est: dict[str, float], p_true: dict[str, float]) -> float:
    total = sum((abs(p_est[s] - p_true[s]) for s in p_true.keys()))
    return 0.5 * total

def generate_plot_1_convergence() -> None:
    bn = student_network()
    query = 'FinalGrade'
    evidence = {'StudyTime': 'Low', 'Attendance': 'Good'}
    exact_res = variable_elimination(bn, query, evidence)
    sample_sizes = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]
    rej_errors = []
    lw_errors = []
    gibbs_errors = []
    for n in sample_sizes:
        r_runs = []
        lw_runs = []
        g_runs = []
        for seed in [10, 20, 30]:
            r_est = rejection_sampling(bn, query, evidence, n_samples=n, seed=seed)
            lw_est = likelihood_weighting(bn, query, evidence, n_samples=n, seed=seed)
            g_est = gibbs_sampling(bn, query, evidence, n_samples=n, burn_in=min(500, n // 2), seed=seed)
            r_runs.append(compute_tvd(r_est, exact_res))
            lw_runs.append(compute_tvd(lw_est, exact_res))
            g_runs.append(compute_tvd(g_est, exact_res))
        rej_errors.append(float(np.mean(r_runs)))
        lw_errors.append(float(np.mean(lw_runs)))
        gibbs_errors.append(float(np.mean(g_runs)))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(sample_sizes, rej_errors, 'o-', label='Rejection Sampling', color='#E74C3C', linewidth=2)
    ax.plot(sample_sizes, lw_errors, 's-', label='Likelihood Weighting', color='#2980B9', linewidth=2)
    ax.plot(sample_sizes, gibbs_errors, '^-', label='Gibbs Sampling (MCMC)', color='#27AE60', linewidth=2)
    theoretical = [0.6 / np.sqrt(n) for n in sample_sizes]
    ax.plot(sample_sizes, theoretical, '--', label='Theoretical $O(1/\\sqrt{N})$', color='#7F8C8D', alpha=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Samples ($N$)', fontsize=12)
    ax.set_ylabel('Total Variation Distance to Exact VE', fontsize=12)
    ax.set_title('Inference Convergence: Error vs. Number of Samples', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    fig.tight_layout()
    output_path = OUTPUT_DIR / '01_convergence_error.png'
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Generated: {output_path}')

def generate_plot_2_rare_evidence() -> None:
    bn = student_network()
    query = 'FinalGrade'
    scenarios = [('Common\n(Att=Good)', {'Attendance': 'Good'}), ('Moderate\n(ST=Low, Att=Poor)', {'StudyTime': 'Low', 'Attendance': 'Poor'}), ('Rare\n(ST=High, PG=Low, Att=Poor)', {'StudyTime': 'High', 'PriorGrade': 'Low', 'Attendance': 'Poor'})]
    labels = [s[0] for s in scenarios]
    rej_rates = []
    lw_retention = []
    n_samples = 20000
    for _, ev in scenarios:
        _, r_diag = rejection_sampling(bn, query, ev, n_samples=n_samples, seed=42, return_diagnostics=True)
        rej_rates.append(r_diag['acceptance_rate'] * 100.0)
        lw_retention.append(100.0)
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5.5))
    rects1 = ax.bar(x - width / 2, rej_rates, width, label='Rejection Sampling (% Kept)', color='#E74C3C')
    rects2 = ax.bar(x + width / 2, lw_retention, width, label='Likelihood Weighting (% Kept)', color='#2980B9')
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontweight='bold')
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Percentage of Samples Retained (%)', fontsize=12)
    ax.set_title('Rare Evidence Stress Test: Rejection vs. Likelihood Weighting', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    output_path = OUTPUT_DIR / '02_rare_evidence.png'
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Generated: {output_path}')

def generate_plot_3_runtime_scaling() -> None:
    chain_sizes_enum = [3, 5, 7, 9, 11, 13, 15]
    chain_sizes_ve = [3, 5, 7, 9, 11, 13, 15, 20, 25, 30]
    enum_times = []
    ve_times = []
    lw_times = []
    for n in chain_sizes_enum:
        bn = chain_network(n)
        query = f'X{n - 1}'
        evidence = {'X0': 's0'}
        start = time.perf_counter()
        enumeration_ask(bn, query, evidence)
        enum_times.append(time.perf_counter() - start)
    for n in chain_sizes_ve:
        bn = chain_network(n)
        query = f'X{n - 1}'
        evidence = {'X0': 's0'}
        start = time.perf_counter()
        variable_elimination(bn, query, evidence)
        ve_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        likelihood_weighting(bn, query, evidence, n_samples=3000, seed=42)
        lw_times.append(time.perf_counter() - start)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(chain_sizes_enum, enum_times, 'o-', label='Enumeration (Exact $O(2^N)$)', color='#E74C3C', linewidth=2.5)
    ax.plot(chain_sizes_ve, ve_times, 's-', label='Variable Elimination (Exact $O(N)$ on chain)', color='#2ECC71', linewidth=2.5)
    ax.plot(chain_sizes_ve, lw_times, '^-', label='Likelihood Weighting (Approx. $N=3000$)', color='#3498DB', linewidth=2.5)
    ax.set_yscale('log')
    ax.set_xlabel('Number of Variables in Chain Network ($N$)', fontsize=12)
    ax.set_ylabel('Runtime in Seconds (Log Scale)', fontsize=12)
    ax.set_title('Computational Scalability: Exact vs. Approximate Inference', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    fig.tight_layout()
    output_path = OUTPUT_DIR / '03_runtime_scaling.png'
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Generated: {output_path}')

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Generating comparative evaluation figures...')
    generate_plot_1_convergence()
    generate_plot_2_rare_evidence()
    generate_plot_3_runtime_scaling()
    print('All figures successfully generated in figures/')
if __name__ == '__main__':
    main()
