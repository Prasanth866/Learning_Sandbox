"""Synthetic data generator for Bayesian Networks.

This module provides tools to generate synthetic data from a Bayesian Network
with known ground-truth CPTs. This is useful for:

1. Verifying parameter learning algorithms (MLE, Bayesian estimation)
2. Testing approximate inference methods against exact inference
3. Running tests without network access
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from network import BayesNet


def default_network() -> BayesNet:
    """Create a small Bayesian Network with known ground-truth CPTs.

    The network structure:
        A -> B -> C
        |    |
        v    v
        D <- E

    Returns
    -------
    BayesNet
        A Bayesian Network with known CPTs.
    """
    bn = BayesNet()

    bn.add_node(
        'A',
        ['a0', 'a1'],
        cpt=np.array([[0.6], [0.4]]),
    )

    bn.add_node(
        'B',
        ['b0', 'b1'],
        parents=['A'],
        cpt=np.array([[0.9, 0.2], [0.1, 0.8]]),
    )

    bn.add_node(
        'C',
        ['c0', 'c1'],
        parents=['B'],
        cpt=np.array([[0.7, 0.3], [0.3, 0.7]]),
    )

    bn.add_node(
        'D',
        ['d0', 'd1'],
        parents=['A'],
        cpt=np.array([[0.8, 0.2], [0.2, 0.8]]),
    )

    bn.add_node(
        'E',
        ['e0', 'e1'],
        parents=['B', 'D'],
        cpt=np.array([
            [[0.9, 0.6], [0.4, 0.7]],
            [[0.1, 0.4], [0.6, 0.3]],
        ]),
    )

    return bn


def sample_synthetic(
    bn: BayesNet,
    n_samples: int,
    seed: int | None = None,
) -> List[Dict[str, str]]:
    """Generate synthetic data by sampling from a Bayesian Network.

    Parameters
    ----------
    bn : BayesNet
        The Bayesian Network to sample from.
    n_samples : int
        Number of samples to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    List[Dict[str, str]]
        List of samples, where each sample is a dictionary mapping
        variable names to state names.
    """
    rng = np.random.default_rng(seed)
    samples = []
    topo_order = bn.topological_sort()

    for _ in range(n_samples):
        sample = {}
        for node in topo_order:
            parents = bn.parents[node]
            parent_states = [sample[p] for p in parents]
            parent_indices = [bn.states[p].index(s) for p, s in zip(parents, parent_states)]

            cpt = bn.cpts[node].values

            if parents:
                idx = tuple(parent_indices)
                probs = cpt[(...,) + idx]
            else:
                probs = cpt.flatten()

            state_idx = rng.choice(len(bn.states[node]), p=probs)
            sample[node] = bn.states[node][state_idx]

        samples.append(sample)

    return samples


def student_network() -> BayesNet:
    """Create a 4-node educational student performance network for slides.

    Graph topology:
        StudyTime -> PriorGrade -> FinalGrade
        Attendance -------------> FinalGrade

    Returns
    -------
    BayesNet
        A 4-node student Bayesian Network.
    """
    bn = BayesNet()

    bn.add_node(
        "StudyTime",
        ["Low", "High"],
        cpt=np.array([[0.4], [0.6]]),
    )

    bn.add_node(
        "Attendance",
        ["Poor", "Good"],
        cpt=np.array([[0.3], [0.7]]),
    )

    bn.add_node(
        "PriorGrade",
        ["Low", "High"],
        parents=["StudyTime"],
        cpt=np.array([
            [0.75, 0.20],
            [0.25, 0.80],
        ]),
    )

    bn.add_node(
        "FinalGrade",
        ["Fail", "Pass"],
        parents=["PriorGrade", "Attendance"],
        cpt=np.array([
            [[0.85, 0.40], [0.35, 0.10]],
            [[0.15, 0.60], [0.65, 0.90]],
        ]),
    )

    return bn


def chain_network(n_nodes: int) -> BayesNet:
    """Create a linear Markov chain network X_0 -> X_1 -> ... -> X_{n-1}.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the chain.

    Returns
    -------
    BayesNet
        A linear chain Bayesian Network.
    """
    bn = BayesNet()
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")

    bn.add_node("X0", ["s0", "s1"], cpt=np.array([[0.6], [0.4]]))

    for i in range(1, n_nodes):
        curr_name = f"X{i}"
        prev_name = f"X{i-1}"
        bn.add_node(
            curr_name,
            ["s0", "s1"],
            parents=[prev_name],
            cpt=np.array([[0.8, 0.2], [0.2, 0.8]]),
        )

    return bn
