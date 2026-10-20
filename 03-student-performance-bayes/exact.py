from __future__ import annotations
import itertools
import numpy as np
from typing import Dict, List, Optional
from network import BayesNet
from factor import Factor

def enumeration_ask(bn: BayesNet, query: str, evidence: Dict[str, str | int]) -> Dict[str, float]:
    if query not in bn.nodes:
        raise ValueError(f'Query variable {query} not in network.')
    int_evidence = {}
    for var, val in evidence.items():
        if var not in bn.nodes:
            raise ValueError(f'Evidence variable {var} not in network.')
        if isinstance(val, str):
            int_evidence[var] = bn.states[var].index(val)
        else:
            int_evidence[var] = val
    all_vars = bn.nodes
    remaining_vars = [var for var in all_vars if var not in int_evidence and var != query]
    assignments = []
    if not remaining_vars:
        assignments = [{}]
    else:
        for assignment in itertools.product(*[range(len(bn.states[var])) for var in remaining_vars]):
            full_assignment = {}
            for var, idx in zip(remaining_vars, assignment):
                full_assignment[var] = idx
            assignments.append(full_assignment)
    query_states = bn.states[query]
    posterior = {state: 0.0 for state in query_states}
    for state_idx, state in enumerate(query_states):
        partial_assignment = {**int_evidence, query: state_idx}
        for assignment in assignments:
            full_assignment = {**partial_assignment, **assignment}
            posterior[state] += bn.joint_probability(full_assignment)
    total = sum(posterior.values())
    if total == 0:
        raise ValueError('Posterior probability sums to zero.')
    for state in query_states:
        posterior[state] /= total
    return posterior

def variable_elimination(bn: BayesNet, query: str, evidence: Dict[str, str | int], order: Optional[List[str]]=None) -> Dict[str, float]:
    if query not in bn.nodes:
        raise ValueError(f'Query variable {query} not in network.')
    int_evidence = {}
    for var, val in evidence.items():
        if var not in bn.nodes:
            raise ValueError(f'Evidence variable {var} not in network.')
        if isinstance(val, str):
            int_evidence[var] = bn.states[var].index(val)
        else:
            int_evidence[var] = val
    all_vars = bn.nodes
    hidden_vars = [var for var in all_vars if var not in int_evidence and var != query]
    if order is None:
        order = min_fill_order(bn, hidden_vars, int_evidence)
    elif not set(order).issubset(set(hidden_vars)):
        raise ValueError('Elimination order contains variables not in hidden non-evidence variables.')
    factors = []
    for var in all_vars:
        f = bn.cpts[var]
        ev = {k: int_evidence[k] for k in f.variables if k in int_evidence}
        if ev:
            f = f.reduce(ev)
        factors.append(f)
    for var in order:
        factors_with_var = [f for f in factors if var in f.variables]
        factors = [f for f in factors if var not in f.variables]
        if not factors_with_var:
            continue
        prod = factors_with_var[0]
        for f in factors_with_var[1:]:
            prod = prod.product(f)
        new_f = prod.marginalize([var])
        if new_f.variables:
            factors.append(new_f)
    if not factors:
        return {s: 1.0 / len(bn.states[query]) for s in bn.states[query]}
    final_factor = factors[0]
    for f in factors[1:]:
        final_factor = final_factor.product(f)
    final_factor = final_factor.normalize()
    query_states = bn.states[query]
    posterior = {query_states[i]: float(final_factor.values[i]) for i in range(len(query_states))}
    return posterior

def min_fill_order(bn: BayesNet, variables: List[str], evidence: Dict[str, int]) -> List[str]:
    remaining_vars = variables.copy()
    order = []
    while remaining_vars:
        min_fill = -1
        best_var = None
        for var in remaining_vars:
            neighbors = set()
            neighbors.update(bn.parents[var])
            neighbors.update(bn.children[var])
            for child in bn.children[var]:
                neighbors.update(bn.parents[child])
            neighbors = [n for n in neighbors if n in remaining_vars and n not in evidence]
            fill = len(neighbors)
            if best_var is None or fill < min_fill:
                min_fill = fill
                best_var = var
        order.append(best_var)
        remaining_vars.remove(best_var)
    return order
variable_elimination.min_fill_order = min_fill_order

def create_evidence_factor(bn: BayesNet, var: str, state_idx: int) -> Factor:
    values = np.zeros(len(bn.states[var]))
    values[state_idx] = 1.0
    return Factor([var], [len(bn.states[var])], values)
