"""Factor class for discrete factors in a Bayesian Network.

A factor is a function from a set of discrete variables to a real number.
It is represented as a multidimensional array (numpy.ndarray) where each
dimension corresponds to a variable and the size of the dimension is the
number of states (cardinality) of that variable.

The factor supports operations needed for exact inference:
- product: pointwise product of two factors
- marginalize: sum out a set of variables
- reduce: condition on evidence (set variables to specific values)
- normalize: normalize the factor so that it sums to 1
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


class Factor:
    """Discrete factor represented as a numpy array with named axes."""

    def __init__(
        self,
        variables: List[str],
        cardinalities: List[int],
        values: np.ndarray,
    ) -> None:
        """Initialize a factor.

        Parameters
        ----------
        variables : List[str]
            Names of the variables in the factor, in order.
        cardinalities : List[int]
            Number of states for each variable, matching the order of `variables`.
        values : np.ndarray
            Array of shape `cardinalities` containing the factor values.

        Raises
        ------
        ValueError
            If the length of `variables` and `cardinalities` do not match,
            or if the shape of `values` does not match `cardinalities`.
        """
        if len(variables) != len(cardinalities):
            raise ValueError(
                f"Length of variables ({len(variables)}) must match length of cardinalities ({len(cardinalities)})"
            )
        if values.shape != tuple(cardinalities):
            raise ValueError(
                f"Shape of values {values.shape} does not match cardinalities {tuple(cardinalities)}"
            )

        self.variables: List[str] = variables
        self.cardinalities: List[int] = cardinalities
        self.values: np.ndarray = values.copy()

    def __repr__(self) -> str:
        """Return a string representation of the factor."""
        return (
            f"Factor(variables={self.variables}, "
            f"cardinalities={self.cardinalities}, "
            f"values_shape={self.values.shape})"
        )

    def __str__(self) -> str:
        """Return a readable string representation of the factor."""
        return self.__repr__()

    def _validate_variables(self, vars_to_check: List[str]) -> None:
        """Check that all variables in vars_to_check are in this factor."""
        for var in vars_to_check:
            if var not in self.variables:
                raise ValueError(f"Variable {var} not in factor {self.variables}")

    def _get_variable_indices(self, vars_of_interest: List[str]) -> List[int]:
        """Return the indices of vars_of_interest in self.variables."""
        return [self.variables.index(var) for var in vars_of_interest]

    def product(self, other: Factor) -> Factor:
        """Compute the pointwise product of two factors.

        The resulting factor has variables that are the union of the variables
        of the two factors. If a variable appears in both, it appears once in
        the result.

        Parameters
        ----------
        other : Factor
            The factor to multiply with.

        Returns
        -------
        Factor
            A new factor representing the product.
        """
        new_vars = self.variables.copy()
        for var in other.variables:
            if var not in new_vars:
                new_vars.append(var)

        new_card = []
        var_to_card = dict(zip(self.variables, self.cardinalities))
        var_to_card.update(dict(zip(other.variables, other.cardinalities)))
        for var in new_vars:
            new_card.append(var_to_card[var])

        v1 = self.values
        curr_vars1 = list(self.variables)
        for v in new_vars:
            if v not in curr_vars1:
                v1 = np.expand_dims(v1, axis=-1)
                curr_vars1.append(v)
        perm1 = [curr_vars1.index(v) for v in new_vars]
        v1_aligned = np.transpose(v1, perm1)

        v2 = other.values
        curr_vars2 = list(other.variables)
        for v in new_vars:
            if v not in curr_vars2:
                v2 = np.expand_dims(v2, axis=-1)
                curr_vars2.append(v)
        perm2 = [curr_vars2.index(v) for v in new_vars]
        v2_aligned = np.transpose(v2, perm2)

        prod_values = v1_aligned * v2_aligned
        return Factor(new_vars, new_card, prod_values)

    def marginalize(self, vars_to_marginalize: List[str]) -> Factor:
        """Sum out the specified variables from the factor.

        Parameters
        ----------
        vars_to_marginalize : List[str]
            Variables to eliminate from the factor.

        Returns
        -------
        Factor
            A new factor with the variables summed out.
        """
        self._validate_variables(vars_to_marginalize)

        remaining_vars = [var for var in self.variables if var not in vars_to_marginalize]

        if not remaining_vars:
            total = self.values.sum()
            return Factor([], [], np.array(total).reshape(()))

        marginalize_idx = [self.variables.index(var) for var in vars_to_marginalize]

        marginalized_values = self.values.sum(axis=tuple(marginalize_idx))

        remaining_card = [
            self.cardinalities[self.variables.index(var)] for var in remaining_vars
        ]

        return Factor(remaining_vars, remaining_card, marginalized_values)

    def reduce(self, evidence: Dict[str, int]) -> Factor:
        """Condition on evidence by setting variables to specific values.

        Parameters
        ----------
        evidence : Dict[str, int]
            Mapping from variable name to integer state index.

        Returns
        -------
        Factor
            A new factor with the evidence variables removed (since they are fixed)
            and the values sliced accordingly.
        """
        self._validate_variables(list(evidence.keys()))

        remaining_vars = [var for var in self.variables if var not in evidence]

        if not remaining_vars:
            idx = tuple(evidence[var] for var in self.variables)
            scalar_value = self.values[idx]
            return Factor([], [], np.array(scalar_value).reshape(()))

        slice_tuple = []
        for var in self.variables:
            if var in evidence:
                slice_tuple.append(evidence[var])
            else:
                slice_tuple.append(slice(None))

        reduced_values = self.values[tuple(slice_tuple)]

        remaining_card = [
            self.cardinalities[self.variables.index(var)] for var in remaining_vars
        ]

        return Factor(remaining_vars, remaining_card, reduced_values)

    def normalize(self) -> Factor:
        """Normalize the factor so that the sum of its values is 1.

        Returns
        -------
        Factor
            A new normalized factor (same variables and cardinalities).
        """
        total = self.values.sum()
        if total == 0:
            raise ValueError("Cannot normalize a factor that sums to zero.")
        normalized_values = self.values / total
        return Factor(self.variables, self.cardinalities, normalized_values)

    def __call__(self, assignment: Dict[str, int]) -> float:
        """Evaluate the factor for a given assignment of variables.

        Parameters
        ----------
        assignment : Dict[str, int]
            Mapping from variable name to integer state index.

        Returns
        -------
        float
            The factor value for the given assignment.

        Raises
        ------
        ValueError
            If the assignment does not match the factor's variables.
        """
        if set(assignment.keys()) != set(self.variables):
            raise ValueError(
                f"Assignment must contain exactly the variables {self.variables}, got {set(assignment.keys())}"
            )

        idx = tuple(assignment[var] for var in self.variables)
        return float(self.values[idx])