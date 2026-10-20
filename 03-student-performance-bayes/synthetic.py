from __future__ import annotations
import numpy as np
from typing import Dict, List
from network import BayesNet

def default_network() -> BayesNet:
    bn = BayesNet()
    bn.add_node('A', ['a0', 'a1'], cpt=np.array([[0.6], [0.4]]))
    bn.add_node('B', ['b0', 'b1'], parents=['A'], cpt=np.array([[0.9, 0.2], [0.1, 0.8]]))
    bn.add_node('C', ['c0', 'c1'], parents=['B'], cpt=np.array([[0.7, 0.3], [0.3, 0.7]]))
    bn.add_node('D', ['d0', 'd1'], parents=['A'], cpt=np.array([[0.8, 0.2], [0.2, 0.8]]))
    bn.add_node('E', ['e0', 'e1'], parents=['B', 'D'], cpt=np.array([[[0.9, 0.6], [0.4, 0.7]], [[0.1, 0.4], [0.6, 0.3]]]))
    return bn

def sample_synthetic(bn: BayesNet, n_samples: int, seed: int | None=None) -> List[Dict[str, str]]:
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
    bn = BayesNet()
    bn.add_node('StudyTime', ['Low', 'High'], cpt=np.array([[0.4], [0.6]]))
    bn.add_node('Attendance', ['Poor', 'Good'], cpt=np.array([[0.3], [0.7]]))
    bn.add_node('PriorGrade', ['Low', 'High'], parents=['StudyTime'], cpt=np.array([[0.75, 0.2], [0.25, 0.8]]))
    bn.add_node('FinalGrade', ['Fail', 'Pass'], parents=['PriorGrade', 'Attendance'], cpt=np.array([[[0.85, 0.4], [0.35, 0.1]], [[0.15, 0.6], [0.65, 0.9]]]))
    return bn

def chain_network(n_nodes: int) -> BayesNet:
    bn = BayesNet()
    if n_nodes < 1:
        raise ValueError('n_nodes must be >= 1')
    bn.add_node('X0', ['s0', 's1'], cpt=np.array([[0.6], [0.4]]))
    for i in range(1, n_nodes):
        curr_name = f'X{i}'
        prev_name = f'X{i - 1}'
        bn.add_node(curr_name, ['s0', 's1'], parents=[prev_name], cpt=np.array([[0.8, 0.2], [0.2, 0.8]]))
    return bn
