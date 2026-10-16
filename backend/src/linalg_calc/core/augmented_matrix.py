from fractions import Fraction

from .matrix import Matrix


class AugmentedMatrix(Matrix):

    def __init__(self, aug_rows: int, aug_cols: int, aug_data: list[list[Fraction]]) -> None:
        super(rows=aug_rows, cols=aug_cols, data=aug_data)

    