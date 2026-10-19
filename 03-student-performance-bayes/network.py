"""Bayesian Network class for representing discrete Bayesian Networks.

A Bayesian Network is a directed acyclic graph (DAG) where each node represents
a discrete random variable and has an associated Conditional Probability Table (CPT).
The joint distribution factorizes as:
    P(X_1, ..., X_n) = \\prod_{i=1}^n P(X_i | Parents(X_i))
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Set, Tuple

from factor import Factor


class BayesNet:
    """Discrete Bayesian Network with conditional probability tables."""

    def __init__(self) -> None:
        """Initialize an empty Bayesian Network."""
        self.nodes: List[str] = []
        self.parents: Dict[str, List[str]] = {}
        self.children: Dict[str, List[str]] = {}
        self.states: Dict[str, List[str]] = {}
        self.cpts: Dict[str, Factor] = {}

    def add_node(
        self,
        name: str,
        states: List[str],
        parents: List[str] | None = None,
        cpt: np.ndarray | None = None,
    ) -> None:
        """Add a node with states, parents, and CPT to the network.

        Parameters
        ----------
        name : str
            Name of the random variable.
        states : List[str]
            List of state names for this variable.
        parents : List[str], optional
            Names of the parent variables in the DAG.
        cpt : np.ndarray, optional
            Conditional probability table. The shape must be
            `[len(states)] + [len(self.states[p]) for p in parents]`.
            The sum along axis 0 (the variable itself) must equal 1 for all parent assignments.
        """
        if name in self.nodes:
            raise ValueError(f"Node {name} already exists in the network.")

        parents = parents or []
        for p in parents:
            if p not in self.nodes:
                raise ValueError(f"Parent {p} must be added to the network before node {name}.")

        self.nodes.append(name)
        self.parents[name] = parents
        self.children[name] = []
        for p in parents:
            self.children[p].append(name)

        self.states[name] = states

        if cpt is not None:
            self.set_cpt(name, cpt)

    def set_cpt(self, name: str, cpt: np.ndarray) -> None:
        """Set or update the CPT for an existing node.

        Parameters
        ----------
        name : str
            Name of the variable.
        cpt : np.ndarray
            The CPT values.
        """
        if name not in self.nodes:
            raise ValueError(f"Node {name} not found in the network.")

        parents = self.parents[name]
        expected_vars = [name] + parents
        expected_cards = [len(self.states[name])] + [len(self.states[p]) for p in parents]

        if len(parents) == 0 and cpt.ndim == 2 and cpt.shape[1] == 1:
            cpt = cpt.flatten()

        if cpt.shape != tuple(expected_cards):
            raise ValueError(
                f"CPT shape {cpt.shape} does not match expected shape {tuple(expected_cards)} for {expected_vars}"
            )

        sums = cpt.sum(axis=0)
        if not np.allclose(sums, 1.0):
            raise ValueError(f"CPT for node {name} does not sum to 1 along its states axis.")

        self.cpts[name] = Factor(expected_vars, expected_cards, cpt)

    def topological_sort(self) -> List[str]:
        """Return a topological ordering of the nodes.

        Returns
        -------
        List[str]
            Topologically ordered node names.
        """
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
            raise ValueError("The graph contains a cycle! Not a valid DAG.")

        return order

    def joint_probability(self, assignment: Dict[str, str | int]) -> float:
        """Compute the joint probability P(X_1=x_1, ..., X_n=x_n).

        Parameters
        ----------
        assignment : Dict[str, str | int]
            Full assignment of all variables to either state names or integer indices.

        Returns
        -------
        float
            Joint probability of the full assignment.
        """
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
        """Compute the Markov blanket of a node.

        The Markov blanket consists of:
        - The node's parents
        - The node's children
        - Other parents of the node's children (spouses)

        Parameters
        ----------
        node : str
            The variable name.

        Returns
        -------
        Set[str]
            Set of variable names forming the Markov blanket.
        """
        if node not in self.nodes:
            raise ValueError(f"Node {node} not in network.")

        mb = set(self.parents[node])
        for child in self.children[node]:
            mb.add(child)
            for parent in self.parents[child]:
                if parent != node:
                    mb.add(parent)

        return mb
