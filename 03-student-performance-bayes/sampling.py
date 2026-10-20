from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Tuple
from network import BayesNet

def rejection_sampling(bn: BayesNet, query: str, evidence: Dict[str, str | int], n_samples: int=10000, seed: int | None=None, return_diagnostics: bool=False) -> Dict[str, float] | Tuple[Dict[str, float], Dict[str, Any]]:
    if query not in bn.nodes:
        raise ValueError(f'Query variable {query} not in network.')
    int_evidence = {}
    for var, val in evidence.items():
        if var not in bn.nodes:
            raise ValueError(f'Evidence variable {var} not in network.')
        if isinstance(val, str):
            int_evidence[var] = bn.states[var].index(val)
        else:
            int_evidence[var] = int(val)
    rng = np.random.default_rng(seed)
    topo_order = bn.topological_sort()
    query_states = bn.states[query]
    counts = {s: 0 for s in query_states}
    accepted = 0
    history = []
    for _ in range(n_samples):
        sample: Dict[str, int] = {}
        for node in topo_order:
            parents = bn.parents[node]
            cpt = bn.cpts[node].values
            if parents:
                parent_indices = tuple((sample[p] for p in parents))
                probs = cpt[(...,) + parent_indices]
            else:
                probs = cpt.flatten()
            chosen_state = int(rng.choice(len(bn.states[node]), p=probs))
            sample[node] = chosen_state
        matches_evidence = True
        for e_var, e_val in int_evidence.items():
            if sample[e_var] != e_val:
                matches_evidence = False
                break
        if matches_evidence:
            state_name = query_states[sample[query]]
            counts[state_name] += 1
            accepted += 1
        if return_diagnostics and accepted > 0:
            current_est = {s: counts[s] / accepted for s in query_states}
            history.append(current_est)
    if accepted > 0:
        posterior = {s: counts[s] / accepted for s in query_states}
    else:
        posterior = {s: 1.0 / len(query_states) for s in query_states}
    if not return_diagnostics:
        return posterior
    diagnostics = {'n_samples': n_samples, 'accepted': accepted, 'rejected': n_samples - accepted, 'acceptance_rate': accepted / n_samples if n_samples > 0 else 0.0, 'history': history}
    return (posterior, diagnostics)

def likelihood_weighting(bn: BayesNet, query: str, evidence: Dict[str, str | int], n_samples: int=10000, seed: int | None=None, return_diagnostics: bool=False) -> Dict[str, float] | Tuple[Dict[str, float], Dict[str, Any]]:
    if query not in bn.nodes:
        raise ValueError(f'Query variable {query} not in network.')
    int_evidence = {}
    for var, val in evidence.items():
        if var not in bn.nodes:
            raise ValueError(f'Evidence variable {var} not in network.')
        if isinstance(val, str):
            int_evidence[var] = bn.states[var].index(val)
        else:
            int_evidence[var] = int(val)
    rng = np.random.default_rng(seed)
    topo_order = bn.topological_sort()
    query_states = bn.states[query]
    weights = {s: 0.0 for s in query_states}
    sample_weights: List[float] = []
    history = []
    for _ in range(n_samples):
        sample: Dict[str, int] = {}
        w = 1.0
        for node in topo_order:
            parents = bn.parents[node]
            cpt = bn.cpts[node].values
            if parents:
                parent_indices = tuple((sample[p] for p in parents))
                probs = cpt[(...,) + parent_indices]
            else:
                probs = cpt.flatten()
            if node in int_evidence:
                obs_val = int_evidence[node]
                sample[node] = obs_val
                w *= float(probs[obs_val])
            else:
                chosen_state = int(rng.choice(len(bn.states[node]), p=probs))
                sample[node] = chosen_state
        state_name = query_states[sample[query]]
        weights[state_name] += w
        sample_weights.append(w)
        if return_diagnostics:
            tot_w = sum(weights.values())
            if tot_w > 0:
                current_est = {s: weights[s] / tot_w for s in query_states}
            else:
                current_est = {s: 1.0 / len(query_states) for s in query_states}
            history.append(current_est)
    total_weight = sum(weights.values())
    if total_weight > 0:
        posterior = {s: weights[s] / total_weight for s in query_states}
    else:
        posterior = {s: 1.0 / len(query_states) for s in query_states}
    if not return_diagnostics:
        return posterior
    w_arr = np.array(sample_weights)
    sum_sq = np.sum(w_arr ** 2)
    ess = float(np.sum(w_arr) ** 2 / sum_sq) if sum_sq > 0 else 0.0
    diagnostics = {'n_samples': n_samples, 'total_weight': total_weight, 'effective_sample_size': ess, 'history': history}
    return (posterior, diagnostics)

def gibbs_sampling(bn: BayesNet, query: str, evidence: Dict[str, str | int], n_samples: int=10000, burn_in: int=1000, thin: int=1, seed: int | None=None, return_diagnostics: bool=False) -> Dict[str, float] | Tuple[Dict[str, float], Dict[str, Any]]:
    if query not in bn.nodes:
        raise ValueError(f'Query variable {query} not in network.')
    int_evidence = {}
    for var, val in evidence.items():
        if var not in bn.nodes:
            raise ValueError(f'Evidence variable {var} not in network.')
        if isinstance(val, str):
            int_evidence[var] = bn.states[var].index(val)
        else:
            int_evidence[var] = int(val)
    rng = np.random.default_rng(seed)
    query_states = bn.states[query]
    non_evidence = [v for v in bn.nodes if v not in int_evidence]
    current_state: Dict[str, int] = dict(int_evidence)
    for v in non_evidence:
        current_state[v] = int(rng.integers(0, len(bn.states[v])))
    counts = {s: 0 for s in query_states}
    recorded_samples = 0
    history = []
    total_steps = burn_in + n_samples * thin
    for step in range(total_steps):
        for var in non_evidence:
            card = len(bn.states[var])
            probs = np.zeros(card, dtype=float)
            parents = bn.parents[var]
            cpt_var = bn.cpts[var].values
            children = bn.children[var]
            for val_idx in range(card):
                current_state[var] = val_idx
                if parents:
                    p_idx = tuple((current_state[p] for p in parents))
                    p_var = float(cpt_var[(val_idx,) + p_idx])
                else:
                    p_var = float(cpt_var[val_idx])
                p_children = 1.0
                for ch in children:
                    ch_parents = bn.parents[ch]
                    ch_cpt = bn.cpts[ch].values
                    ch_idx = current_state[ch]
                    ch_p_idx = tuple((current_state[p] for p in ch_parents))
                    p_children *= float(ch_cpt[(ch_idx,) + ch_p_idx])
                probs[val_idx] = p_var * p_children
            total_p = probs.sum()
            if total_p > 0:
                probs /= total_p
            else:
                probs = np.ones(card) / card
            new_val = int(rng.choice(card, p=probs))
            current_state[var] = new_val
        if step >= burn_in and (step - burn_in) % thin == 0:
            state_name = query_states[current_state[query]]
            counts[state_name] += 1
            recorded_samples += 1
            if return_diagnostics:
                current_est = {s: counts[s] / recorded_samples for s in query_states}
                history.append(current_est)
    if recorded_samples > 0:
        posterior = {s: counts[s] / recorded_samples for s in query_states}
    else:
        posterior = {s: 1.0 / len(query_states) for s in query_states}
    if not return_diagnostics:
        return posterior
    diagnostics = {'n_samples': recorded_samples, 'burn_in': burn_in, 'thin': thin, 'history': history}
    return (posterior, diagnostics)
