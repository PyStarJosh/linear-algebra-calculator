from fractions import Fraction

from .matrix import Matrix


class AugmentedMatrix(Matrix):
    def __init__(
        self, aug_rows: int, aug_cols: int, aug_data: list[list[Fraction]]
    ) -> None:
        super().__init__(aug_rows, aug_cols, aug_data)
        self._augment_col = self._cols - 1

    def __repr__(self) -> str:
        return f"AugmentedMatrix(rows={self._rows!r}, cols={self._cols!r}, data={self._data!r})"

    def __str__(self) -> str:
        entries: list[str] = [[str(cell) for cell in row] for row in self._data]
        widths: list[int] = [
            max(len(entries[row][col]) for row in range(self._rows))
            for col in range(self._cols)
        ]

        lines: list[str] = [
            "[ "
            + "  ".join(
                f"| {entry.rjust(widths[col_idx])}" if col_idx == self._augment_col 
                else entry.rjust(widths[col_idx]) for col_idx, entry in enumerate(row)
            )
            + " ]"
            for row in entries
        ]

        return "\n".join(lines)

    