from fractions import Fraction

from .base_matrix import BaseMatrix
from .matrix import Matrix


class AugmentedMatrix(BaseMatrix):
    """A linear system consisting of a coefficient matrix and a resultant vector.

    Args:
        BaseMatrix (class): An abstract class with needed attributes and methods.
    """
    def __init__(self, aug_data: list[list[Fraction]]) -> None:
        super().__init__(data=aug_data)
        self._aug_col = self.cols - 1

    @property
    def coefficient_matrix(self) -> "Matrix":
        """Truncates the resultant vector from augmented matrix.

        Returns:
            Matrix: The coefficient matrix.
        """
        coefficient_data: list[Fraction] = [[row[entry_idx]] for entry_idx in range(self._aug_col) for row in self.data]
        return Matrix(data=coefficient_data)    

    def __repr__(self) -> str:
        return f"AugmentedMatrix(data={self.data!r})"

    def __str__(self) -> str:
        entries: list[str] = [[str(cell) for cell in row] for row in self.data]
        widths: list[int] = [
            max(len(entries[row][col]) for row in range(self.rows))
            for col in range(self.cols)
        ]

        lines: list[str] = [
            "[ "
            + "  ".join(
                f"| {entry.rjust(widths[col_idx])}" if col_idx == self._aug_col 
                else entry.rjust(widths[col_idx]) for col_idx, entry in enumerate(row)
            )
            + " ]"
            for row in entries
        ]

        return "\n".join(lines)

    def __eq__(self, other: "AugmentedMatrix") -> bool:
        result = super().__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return self.data == other.data
