from abc import ABC, abstractmethod
from fractions import Fraction


class BaseMatrix(ABC):
    """A abstract class consisting of the base attributes and methods matrices needed.

    Args:
        ABC (class): A class that enforces abstraction rules for abstract classes and methods.
    """
    def __init__(self, data: list[list[Fraction]]):
        self._data = data

    @property
    def rows(self) -> int:
        return len(self._data)

    @property
    def cols(self) -> int:
        return len(self._data[0])

    @property
    def data(self) -> list[list[Fraction]]:
        return self._data

    def __eq__(self, other: object) -> bool:
        """Evaluates the equivalence of two matrices.

        Args:
            other (Matrix): A second matrix used for comparison.

        Returns:
            bool: Whether the two matrices are equivalent.
            NotImplemented: If the matrices are not the same type.
        """
        if type(self) is not type(other):
            return NotImplemented
        return self._data == other._data

    @abstractmethod
    def rref(self):
        pass

    @abstractmethod
    def ref(self):
        pass

    # REF AND RREF HELPER METHODS

    def _replacement(
            self, scalar: Fraction, sm_row: list[int], res_row: list[int]
        ) -> list[int]:
            """Takes a scalar multiplied row and adds it to another to row reduce.
    
            Args:
                scalar (Fraction): A scalar value used for multiplying
                sm_row (list[int]): Scalar multiplied row.
                res_row (list[int]): Row being added to and replaced.
    
            Returns:
                list[int]: Updated row reduced row.
            """
            scaled_row = [scalar * entry for entry in sm_row]
            return [scaled_row[idx] + res_row[idx] for idx in range(self.cols)]
    
    def _scale(self, scalar: Fraction, row: list[int]) -> list[int]:
        """Scales a row via scalar multiplication

        Args:
            scalar (Fraction): A scalar value used for scalar multiplication
            row (list[int]): List containing entries to be multiplied

        Returns:
            list[int]: Product list with scaled entries.
        """
        return [scalar * entry for entry in row]

    def _interchange(
        self, top_row: list[int], low_row: list[int]
    ) -> tuple[list[int], list[int]]:
        """_summary_

        Args:
            top_row (list[int]): Row to be lower via interchange
            low_row (list[int]): Row to be risen via interchange.

        Returns:
            tuple[list[int], list[int]]: The two input rows swapped.
        """
        return low_row, top_row