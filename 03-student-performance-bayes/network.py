from __future__ import annotations
import numpy as np
from typing import Dict, List, Set
from factor import Factor

class BayesNet:

    def __init__(self) -> None:
        self.nodes: List[str] = []
        self.parents: Dict[str, List[str]] = {}
        self.children: Dict[str, List[str]] = {}
        self.states: Dict[str, List[str]] = {}
        self.cpts: Dict[str, Factor] = {}

    def add_node(self, name: str, states: List[str], parents: List[str] | None=None, cpt: np.ndarray | None=None) -> None:
        if name in self.nodes:
            raise ValueError(f'Node {name} already exists in the network.')
        parents = parents or []
        for p in parents:
            if p not in self.nodes:
                raise ValueError(f'Parent {p} must be added to the network before node {name}.')
        self.nodes.append(name)
        self.parents[name] = parents
        self.children[name] = []
        for p in parents:
            self.children[p].append(name)
        self.states[name] = states
        if cpt is not None:
            self.set_cpt(name, cpt)

    def set_cpt(self, name: str, cpt: np.ndarray) -> None:
        if name not in self.nodes:
            raise ValueError(f'Node {name} not found in the network.')
        parents = self.parents[name]
        expected_vars = [name] + parents
        expected_cards = [len(self.states[name])] + [len(self.states[p]) for p in parents]
        if len(parents) == 0 and cpt.ndim == 2 and (cpt.shape[1] == 1):
            cpt = cpt.flatten()
        if cpt.shape != tuple(expected_cards):
            raise ValueError(f'CPT shape {cpt.shape} does not match expected shape {tuple(expected_cards)} for {expected_vars}')
        sums = cpt.sum(axis=0)
        if not np.allclose(sums, 1.0):
            raise ValueError(f'CPT for node {name} does not sum to 1 along its states axis.')
        self.cpts[name] = Factor(expected_vars, expected_cards, cpt)

    def topological_sort(self) -> List[str]:
        in_degree = {node: len(self.parents[node]) for node in self.nodes}
        queue = [node for node in self.nodes if in_degree[node] == 0]
        order = []
        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for child in self.children[curr]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if len(order) != len(self.nodes):
            raise ValueError('The graph contains a cycle! Not a valid DAG.')
        return order

    def joint_probability(self, assignment: Dict[str, str | int]) -> float:
        int_assignment = {}
        for node in self.nodes:
            val = assignment[node]
            if isinstance(val, str):
                int_assignment[node] = self.states[node].index(val)
            else:
                int_assignment[node] = int(val)
        prob = 1.0
        for node in self.nodes:
            factor = self.cpts[node]
            node_assignment = {var: int_assignment[var] for var in factor.variables}
            prob *= factor(node_assignment)
        return prob

    def markov_blanket(self, node: str) -> Set[str]:
        if node not in self.nodes:
            raise ValueError(f'Node {node} not in network.')
        mb = set(self.parents[node])
        for child in self.children[node]:
            mb.add(child)
            for parent in self.parents[child]:
                if parent != node:
                    mb.add(parent)
        return mb
