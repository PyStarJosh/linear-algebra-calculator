"""Currently, A sandbox for testing Matrix output on rudimentary use cases. Actual testing in backend/tests."""

from linalg_calc.core.matrix import Matrix

A = Matrix(rows=4, cols=5, data=[
    [2, -3, 0, 0, -9],
    [0, 0, 3,0, 4],
    [0, 0, 0, 1, 3],
    [0, 0, 0, 0, 0]
    ]
)

P = Matrix(rows=3, cols=1, data=[[4], [13], [-4]])

print(A.rref())
