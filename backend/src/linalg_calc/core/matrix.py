"""Matrix attributes and operations for the linear algebra calculator.

This module provides the Matrix class and functions for common matrix operations.
"""

from fractions import Fraction

from .base_matrix import BaseMatrix


class Matrix(BaseMatrix):
    """A 2D matrix of real numbers."""

    def __str__(self) -> str:
        # prepare cells for length tracking
        cells: list[list[str]] = [[str(entry) for entry in row] for row in self.data]

        # find longest numerical value to set right adjustment value for column
        widths: list[int] = [
            max(len(cells[row][col]) for row in range(self.rows))
            for col in range(self.cols)
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
        return f"Matrix(data={self.data!r})"

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
            raise TypeError("A matrix can only be added to another matrix.")

        if (self.rows, self.cols) != (other.rows, other.cols):
            raise ValueError("Matrices must have the same dimensions for addition.")

        total_data = [
            [cell + entry for cell, entry in zip(x, y)]
            for x, y in zip(self.data, other.data)
        ]

        return Matrix(data=total_data)

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
            raise TypeError("An matrix can only be subtracted by another matrix.")

        if (self.rows, self.cols) != (other.rows, other.cols):
            raise ValueError("Matrices must have the same dimensions for subtraction.")

        diff_data = [
            [cell - entry for cell, entry in zip(x, y)]
            for x, y in zip(self.data, other.data)
        ]

        return Matrix(data=diff_data)

    def __mul__(self, scalar: float) -> "Matrix":
        """Computes scalar multiplication of a matrix.

        Args:
            scalar (float): A scalar value multiplied to matrix.

        Returns:
            Matrix: A new matrix with each entry multiplied by passed scalar value.
        """
        smult_data = [[Fraction(scalar) * entry for entry in row] for row in self.data]
        return Matrix(data=smult_data)

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
            return NotImplemented(f"'{type(other)}' has no matrix multiplication method.")
        if self.cols != other.rows:
            raise ValueError("The multiplicand matrix's columns must equal the multiplier's number of rows and vice versa.")
        product_data = [[] for _ in range(self.rows)]  # structures product matrix rows

        for i in range(self.rows):
            for j in range(other.cols):
                res = 0
                for k in range(other.rows):
                    res += self.data[i][k] * other.data[k][j]
                product_data[i].append(res)

        return Matrix(rows=self.rows, cols=other.cols, data=product_data)

    def __pow__(self, exponent: int) -> "Matrix":
        """Computes exponent of matrix

        Args:
            exponent (int): The power the matrix is raised by.

        Returns:
            Matrix: The resulting matrix of matrix^exponent.
        """
        if self.rows != self.cols:
            raise ValueError("The multiplicand matrix's columns must equal the multiplier's number of rows and vice versa.")

        if exponent < 0:
            raise ValueError("Exponents must equal to or greater than 0.")
        elif exponent == 0:
            return self.identity()
        elif exponent == 1:
            return Matrix(
                rows=self.rows, cols=self.cols, data=[row[:] for row in self.data]
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
        id_data = [[] for _ in range(self.rows)]
        for row in range(self.rows):
            for col in range(self.cols):
                if row == col:
                    id_data[row].append(1)
                else:
                    id_data[row].append(0)

        return Matrix(data=id_data)

    def ref(self) -> "Matrix":
        """Computes a row echelon form of the matrix via Gaussian elimination.

        Returns:
            Matrix: A row equivalent matrix in row echelon form.
        """
        copy_matrix = [[Fraction(entry) for entry in row] for row in self.data]
        pivot_row = 0  # row the next pivot will be placed in

        for col in range(self.cols):
            if pivot_row >= self.rows:
                break

            # first row at or below pivot_row with a nonzero entry in this column
            swap_row = next(
                (r for r in range(pivot_row, self.rows) if copy_matrix[r][col] != 0),
                None,
            )
            if swap_row is None:
                continue  #   no pivot in this column, so look in the next column on the same row

            if swap_row != pivot_row:
                copy_matrix[pivot_row], copy_matrix[swap_row] = self._interchange(
                    top_row=copy_matrix[pivot_row],
                    low_row=copy_matrix[swap_row]
                )

            pivot_value = copy_matrix[pivot_row][col]

            for n in range(pivot_row + 1, self.rows):
                curr = copy_matrix[n][col]
                if curr != 0:
                    s_multiplier = (-curr / pivot_value) # multiplier needed to row reduce
                    copy_matrix[n] = self._replacement(
                        scalar=s_multiplier,
                        sm_row=copy_matrix[pivot_row],
                        res_row=copy_matrix[n],
                    )

            pivot_row += 1

        return Matrix(data=copy_matrix)

    def rref(self) -> "Matrix":
        """Computes a reduced row echelon form of the matrix via Gauss-Jordan elimination.

        Returns:
            Matrix: A row equivalent matrix in reduced row echelon form.
        """
        ref_matrix: list[list[Fraction]] = self.ref()._data

        for row in range(self.rows - 1, -1, -1):
            pivot_col = next(
                (c for c in range(self.cols) if ref_matrix[row][c] != 0),
                None
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

        return Matrix(data=ref_matrix)

    def is_invertible(self) -> bool:
        """Determines and returns whether the matrix is invertible or singular.

        Returns:
            bool: A boolean value representing whether the matrix is invertible or singular.
        """            
        return self.rows == self.cols and self.rref() == self.identity()

    def pivot_rows(self):
        rref_matrix = self.rref()._data
        row_pivots = 0
        for row in rref_matrix:
            for col in row:
                if col != 0:
                    row_pivots += 1
                    break
        return row_pivots

    def pivot_cols(self):
        rref_matrix = self.rref()._data
        pivot_cols = sum(1 for row in rref_matrix if any(x != 0 for x in row))
        return pivot_cols

    def dim_col_space(self) -> int:
        """Returns the dimensions of column space of a matrix.

        Returns:
            int: The dimension of column space.
        """
        return self.pivot_rows()

    def dim_null_space(self) -> int:
        """Returns the dimensions of null space of a matrix.

        Returns:
            int: The dimension of null space.
        """
        return self.cols - self.pivot_rows()

    def rank(self) -> int:
        """Returns rank of a matrix

        Returns:
            int: The rank of a matrix.
        """
        return self.pivot_rows()
    
    def spans_rn(self) -> bool:
        """Checks for a pivot in each row of the matrix to determine span of R^n.

        Returns:
            bool: A bool value representing whether the matrix spans Rn.
        """
        row_pivots = self.pivot_cols()
        return row_pivots == self.rows

    def linear_independence(self):
        """Checks for a pivot in each column to determine linear independence of the columns
        Returns:
            bool: A boolean value representing if the matrix is linearly independent.
        """
        pivot_cols = self.pivot_cols()
        return pivot_cols == self.cols