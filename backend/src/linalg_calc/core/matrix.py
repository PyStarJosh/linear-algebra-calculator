"""Matrix attributes and operations for the linear algebra calculator.

This module provides the Matrix class and functions for common matrix operations.
"""

from fractions import Fraction


class Matrix:
    """A 2D matrix of real numbers."""

    def __init__(self, rows: int, cols: int, data: list[list[int]]) -> None:
        """Constructs a 2D matrix.

        Args:
            rows (int): The number of rows in the matrix.
            cols (int): The number of columns in the matrix.
            data (list[list[int]]): The entry values for the matrix.

        Raises:
            ValueError: When rows and columns are set to value less than one.
        """
        if rows <= 0 or cols <= 0:
            raise ValueError("A matrix must have at least one row and column.")
        self._rows = rows
        self._cols = cols
        if self._rows != len(data) or self._cols != len(data[0]):
            raise ValueError(
                f"Inputted matrix entries creates a {len(data)}x{len(data[0])} "
                f"matrix not the provided matrix dimensions: {self._rows}x{self._cols}"
            )
        else:
            self._data = data

    def __str__(self) -> str:
        """Creates a user friendly matrix string for viewing.

        Returns:
            str: A readable formatted str of the martrix.
        """
        # prepare cells for length tracking
        cells: list[list[str]] = [[str(entry) for entry in row] for row in self._data]

        # find longest numerical value to set right adjustment value for column
        widths: list[int] = [
            max(len(cells[row][col]) for row in range(self._rows))
            for col in range(self._cols)
        ]

        # joins rows' list into formatted string
        lines: list[str] = [
            "[ "
            + "  ".join(cell.rjust(width) for cell, width in zip(row, widths))
            + " ]"
            for row in cells
        ]

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Prints a programmer friendly representation of Matrix object.

        Returns:
            str: A string representation of Matrix object initialization
        """
        return f"Matrix(rows={self._rows!r}, cols={self._cols!r}, data={self._data!r})"

    def __eq__(self, other: "Matrix") -> bool:
        """Evaluates the equivalence of two matrices.

        Args:
            other (Matrix): A second matrix used for comparison.

        Returns:
            bool: Whether the two matrices are equivalent.
        """
        return all(
            (
                self._data == other._data,
                self._rows == other._rows,
                self._cols == other._cols,
            )
        )

    def __add__(self, other: "Matrix") -> "Matrix":
        """Computes the sum of two matrices.

        Args:
            other (Matrix): The second matrix needed for addition.

        Raises:
            NotImplementedError: Prevents addition between Matrix and non-Matrix type.
            ValueError: Prevents computation between matrices with different dimensions.

        Returns:
            Matrix: The sum matrix of two matrices.
        """
        if not isinstance(other, Matrix):
            return TypeError("A matrix can only be added to another matrix.")

        if (self._rows, self._cols) != (other._rows, other._cols):
            return ValueError("Matrices must have the same dimensions for addition.")

        total_data = [
            [cell + entry for cell, entry in zip(x, y)]
            for x, y in zip(self._data, other._data)
        ]

        return Matrix(rows=self._rows, cols=self._cols, data=total_data)

    def __sub__(self, other: "Matrix") -> "Matrix":
        """Computes the difference matrix between two matrices.

        Args:
            other (Matrix): Another matrix of same dimensions needed for subtraction.

        Raises:
            NotImplementedError: Prevents subtraction between Matrix and non-Matrix type.
            ValueError: Prevents subtraction between two matrices of different dimensions

        Returns:
            Matrix: The difference matrix of the two matrices.
        """
        if not isinstance(other, Matrix):
            return TypeError("An matrix can only be subtracted by another matrix.")

        if (self._rows, self._cols) != (other._rows, other._cols):
            return ValueError("Matrices must have the same dimensions for subtraction.")

        diff_data = [
            [cell - entry for cell, entry in zip(x, y)]
            for x, y in zip(self._data, other._data)
        ]

        return Matrix(rows=self._rows, cols=self._cols, data=diff_data)

    def __mul__(self, scalar: float) -> "Matrix":
        """Computes scalar multiplication of a matrix.

        Args:
            scalar (float): A scalar value multiplied to matrix.

        Returns:
            Matrix: A new matrix with each entry multiplied by passed scalar value.
        """
        smult_data = [[Fraction(scalar) * entry for entry in row] for row in self._data]
        return Matrix(rows=self._rows, cols=self._cols, data=smult_data)

    def __rmul__(self, scalar: float) -> "Matrix":
        """Allows for communicative scalar multiplication.

        Args:
            scalar (float): A scalar value multiplied to the Matrix's entries.

        Returns:
            Matrix: A matrix with entries multiplied by the scalar.
        """
        return self.__mul__(scalar)

    def __matmul__(self, other: "Matrix") -> "Matrix":
        """Computes product of two matrices.

        Args:
            other (Matrix): The multiplier matrix.

        Raises:
            ValueError: Prevents matrix multiplication where the multiplicand's columns
            do not equal the multiplier's rows.

        Returns:
            Matrix: The product matrix derived from multiplying two matrices.
        """
        if not isinstance(other, Matrix):
            return TypeError(f"'{type(other)}' has no matrix multiplication method.")
        if self._cols != other._rows:
            raise ValueError
        product_data = [[] for _ in range(self._rows)]  # structures product matrix rows

        for i in range(self._rows):
            for j in range(other._cols):
                res = 0
                for k in range(other._rows):
                    res += self._data[i][k] * other._data[k][j]
                product_data[i].append(res)

        return Matrix(rows=self._rows, cols=other._cols, data=product_data)

    def __pow__(self, exponent: int) -> "Matrix":
        """Computes exponent of matrix

        Args:
            exponent (int): The power the matrix is raised by.

        Returns:
            Matrix: The resulting matrix of matrix^exponent.
        """
        if self._rows != self._cols:
            return ValueError

        if exponent < 0:
            return ValueError
        elif exponent == 0:
            return self.identity()
        elif exponent == 1:
            return Matrix(
                rows=self._rows, cols=self._cols, data=self._data[:]
            )  # returns copy of originial matrix
        else:
            res_matrix = self
            for _ in range(exponent - 1):
                res_matrix @= self  # calls __matmul__ to create a new Matrix obj

            return res_matrix

    def identity(self) -> "Matrix":
        """Generates the id matrix of existing matrix.

        Returns:
            Matrix: The identity matrix.
        """
        id_data = [[] for _ in range(self._rows)]
        for row in range(self._rows):
            for col in range(self._cols):
                if row == col:
                    id_data[row].append(1)
                else:
                    id_data[row].append(0)

        return Matrix(rows=self._rows, cols=self._cols, data=id_data)

    def ref(self) -> "Matrix":
        """Computes a row echelon form of the matrix via Gaussian elimination.

        Returns:
            Matrix: A row equivalent matrix in row echelon form.
        """
        copy_matrix = [[Fraction(entry) for entry in row] for row in self._data]
        pivot_row = 0  # row the next pivot will be placed in

        for col in range(self._cols):
            if pivot_row >= self._rows:
                break

            # first row at or below pivot_row with a nonzero entry in this column
            swap_row = next(
                (r for r in range(pivot_row, self._rows) if copy_matrix[r][col] != 0),
                None,
            )
            if swap_row is None:
                continue  #   no pivot in this column, so look in the next column on the same row

            if swap_row != pivot_row:
                copy_matrix[pivot_row], copy_matrix[swap_row] = self._interchange(
                    top_row=copy_matrix[pivot_row], low_row=copy_matrix[swap_row]
                )

            pivot_value = copy_matrix[pivot_row][col]

            for n in range(pivot_row + 1, self._rows):
                curr = copy_matrix[n][col]
                if curr != 0:
                    s_multiplier = (
                        -curr / pivot_value
                    )  # multiplier needed to row reduce
                    copy_matrix[n] = self._replacement(
                        scalar=s_multiplier,
                        sm_row=copy_matrix[pivot_row],
                        res_row=copy_matrix[n],
                    )

            pivot_row += 1

        return Matrix(rows=self._rows, cols=self._cols, data=copy_matrix)

    def rref(self) -> "Matrix":
        """Computes a reduced row echelon form of the matrix via Gauss-Jordan elimination.

        Returns:
            Matrix: A row equivalent matrix in reduced row echelon form.
        """
        ref_matrix: list[list[Fraction]] = self.ref()._data

        for row in range(self._rows - 1, -1, -1):
            pivot_col = next(
                (c for c in range(self._cols) if ref_matrix[row][c] != 0), None
            )

            if pivot_col is None:
                continue

            pivot = ref_matrix[row][pivot_col]

            for r in range(row - 1, -1, -1):
                curr = ref_matrix[r][pivot_col]
                if curr != 0:
                    s_multiplier = -curr / pivot
                    ref_matrix[r] = self._replacement(
                        scalar=s_multiplier,
                        sm_row=ref_matrix[row],
                        res_row=ref_matrix[r],
                    )
            ref_matrix[row] = self._scale(scalar=1 / pivot, row=ref_matrix[row])

        return Matrix(rows=self._rows, cols=self._cols, data=ref_matrix)

    def is_invertible(self) -> bool:
        """Determines and returns whether the matrix is invertible or singular.

        Returns:
            bool: A boolean value representing whether the matrix is invertible or singular.
        """
        return self.rref() == self.identity()

    def spans_Rn(self) -> bool:
        """Checks for a pivot in each row of the matrix to determine span of R^n.

        Returns:
            bool: A bool value representing whether the matrix spans Rn.
        """
        rref_matrix = self.rref()._data
        row_pivots = 0
        for row in rref_matrix:
            for col in row:
                if col != 0:
                    row_pivots += 1
                    break
        return row_pivots == self._rows

    def linear_independence(self):  # IN PROGRESS (Incorrect Output)
        rref_matrix = self.rref()._data
        col_pivots = 0
        for col in range(len(rref_matrix[0])):
            for row in range(len(rref_matrix)):
                if rref_matrix[row][col] != 0:
                    col_pivots += 1
                    break
        return col_pivots == self._cols

    # MATRIX'S HELPER METHODS
    """These are the Matrix's class helper methods used to assist in 
    intermediary computations and various task.
    """

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
        return [scaled_row[idx] + res_row[idx] for idx in range(self._cols)]

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
