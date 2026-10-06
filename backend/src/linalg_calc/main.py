"""Currently, A sandbox for testing Matrix output on rudimentary use cases. Actual testing in backend/tests."""

from linalg_calc.core.matrix import Matrix

A = Matrix(rows=2, cols=3, data=[[1, 1, 6], [1, 1, 2]])

print(A.linear_independence())
