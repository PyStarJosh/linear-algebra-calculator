# Linear Algebra Calculator

A linear algebra calculator written in Python. At its core is a small, dependency-free `Matrix` class that supports the standard matrix operations using ordinary Python operators (`+`, `-`, `*`, `@`, `**`). It can also row-reduce matrices to REF and RREF using exact fraction arithmetic, and answer theory questions such as "is this matrix invertible?" and "do the columns span Rᵐ?". Next up: augmented matrices for solving systems of linear equations, and eventually a web API (FastAPI).

---

## Table of contents

1. [Project status](#project-status)
2. [Getting started](#getting-started)
3. [Project structure](#project-structure)
4. [The `Matrix` class](#the-matrix-class)
5. [Row reduction](#row-reduction)
6. [Matrix properties](#matrix-properties)
7. [Design notes (read this before changing `Matrix`)](#design-notes-read-this-before-changing-matrix)
8. [Known limitations and gotchas](#known-limitations-and-gotchas)
9. [Roadmap](#roadmap)
10. [Testing](#testing)
11. [Coding conventions](#coding-conventions)
12. [Glossary](#glossary)

---

## Project status

| Area                                   | Status                                              |
| -------------------------------------- | --------------------------------------------------- |
| `Matrix` construction and printing     | ✅ Done                                             |
| Equality (`==`)                        | ✅ Done                                             |
| Addition / subtraction                 | ✅ Done (error handling needs a fix, see [gotchas](#known-limitations-and-gotchas)) |
| Scalar multiplication (`A * c`, `c * A`) | ✅ Done                                           |
| Matrix multiplication                  | ✅ Done                                             |
| Integer powers (`A ** k`, k ≥ 0)       | ✅ Done                                             |
| Identity matrix                        | ✅ Done                                             |
| Exact arithmetic with `Fraction`       | ✅ Done (inside `ref()` / `rref()`)                 |
| REF (row echelon form)                 | ✅ Done                                             |
| RREF (reduced row echelon form)        | ✅ Done                                             |
| `spans_Rn()`                           | ✅ Done                                             |
| `is_invertible()`                      | ⚠️ Wrong for some non-square matrices              |
| `linear_independence()`                | 🚧 In progress (gives wrong answers)               |
| Augmented matrices                     | 🗓️ Planned                                          |
| Unit tests                             | 🗓️ Planned (none yet)                               |
| Web API (FastAPI)                      | 🗓️ Planned (dependency declared, no endpoints yet)  |

---

## Getting started

### Prerequisites

- **Python 3.13+** (pinned in `backend/.python-version`)
- **[uv](https://docs.astral.sh/uv/)**, the package and environment manager used by this project

If you don't have uv yet, follow the [install guide](https://docs.astral.sh/uv/getting-started/installation/). You do **not** need to create a virtual environment by hand; uv does that for you.

### Setup

All Python work happens inside the `backend/` directory:

```bash
cd backend
uv sync          # creates .venv/ and installs runtime + dev dependencies from uv.lock
```

### Trying out the code

The quickest way to experiment is an interactive shell:

```bash
uv run python
```

```python
>>> from linalg_calc.core.matrix import Matrix
>>> A = Matrix(rows=2, cols=2, data=[[1, -2], [3, 1]])
>>> print(A ** 3)
[ -17    6 ]
[  -9  -17 ]
>>> print(Matrix(3, 4, [[1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 2, 1]]).rref())
[ 1  0  0     0 ]
[ 0  1  0  -1/2 ]
[ 0  0  1   1/2 ]
```

`src/linalg_calc/main.py` is a sandbox for trying out `Matrix` on quick examples (real tests belong in `backend/tests/`):

```bash
uv run python src/linalg_calc/main.py
```

### Useful commands

| Command                       | What it does                                  |
| ----------------------------- | --------------------------------------------- |
| `uv sync`                     | Install and update dependencies               |
| `uv run pytest`               | Run the test suite                            |
| `uv run ruff check .`         | Lint the code                                 |
| `uv run ruff format .`        | Auto-format the code                          |
| `uv add <package>`            | Add a runtime dependency                      |
| `uv add --dev <package>`      | Add a development-only dependency             |
| `uv run linear-algebra-calculator` | Run the package entry point (currently prints a hello message) |

---

## Project structure

```
linear-algebra-calculator/
├── README.md                 ← you are here
├── .gitignore
└── backend/
    ├── pyproject.toml        ← project metadata, dependencies, entry point
    ├── uv.lock               ← exact locked dependency versions (commit this)
    ├── .python-version       ← Python version used by uv
    ├── src/
    │   └── linalg_calc/
    │       ├── __init__.py   ← package entry point: main()
    │       ├── main.py       ← sandbox script
    │       └── core/
    │           ├── __init__.py
    │           └── matrix.py ← the Matrix class (all the math lives here)
    └── tests/
        └── conftest.py       ← shared pytest fixtures (empty for now)
```

The project uses the **`src/` layout**: the importable package lives in `src/linalg_calc/`, not at the repo root. That means you import it as `linalg_calc...` and should always run code through `uv run`, which installs the package into the virtual environment so the imports resolve.

### Dependencies

| Package              | Type    | Why it's here                                        |
| -------------------- | ------- | ---------------------------------------------------- |
| `fastapi`            | runtime | For the planned web API. Not used yet.               |
| `uvicorn[standard]`  | runtime | ASGI server to run the FastAPI app. Not used yet.    |
| `pytest`             | dev     | Test runner                                          |
| `ruff`               | dev     | Linter and formatter                                 |

The math itself deliberately has **no third-party dependencies** (no NumPy). The only import is `fractions.Fraction` from the standard library. This is a learning-oriented calculator: we implement the algorithms ourselves so we can show and control each step.

---

## The `Matrix` class

**File:** `backend/src/linalg_calc/core/matrix.py`

A `Matrix` stores three private attributes:

| Attribute | Type              | Meaning                                                  |
| --------- | ----------------- | -------------------------------------------------------- |
| `_rows`   | `int`             | Number of rows (must be ≥ 1)                             |
| `_cols`   | `int`             | Number of columns (must be ≥ 1)                          |
| `_data`   | `list[list[...]]` | The entries, stored **row by row** (`_data[i][j]` is row `i`, column `j`). Ints or floats on input; `Fraction`s after `ref()` / `rref()`. |

Indexing is **zero-based** in code, even though math textbooks use one-based indexing. The textbook entry a₁₁ is `_data[0][0]` in code.

### Constructing a matrix

```python
from linalg_calc.core.matrix import Matrix

A = Matrix(
    rows=2,
    cols=3,
    data=[
        [1, 2, 3],
        [4, 5, 6],
    ],
)
```

The constructor raises `ValueError` if `rows` or `cols` is less than 1, or if `data` doesn't match the given dimensions.

### Printing

There are two string forms, following standard Python conventions:

```python
>>> print(A)          # __str__: human-friendly, right-aligned columns
[ 1  2  3 ]
[ 4  5  6 ]

>>> A                 # __repr__: developer-friendly, shows how to rebuild it
Matrix(rows=2, cols=3, data=[[1, 2, 3], [4, 5, 6]])
```

`__str__` measures the widest entry in each column and right-aligns every cell to that width, so negative numbers, fractions like `-1/2` and multi-digit values line up.

### Operations

Every operation returns a **new** `Matrix`. None of them modify the operands. See [Design notes](#design-notes-read-this-before-changing-matrix) for why that matters.

| Operation             | Syntax         | Method        | Requirement                              |
| --------------------- | -------------- | ------------- | ---------------------------------------- |
| Equality              | `A == B`       | `__eq__`      | Same dimensions and entries              |
| Addition              | `A + B`        | `__add__`     | Same dimensions                          |
| Subtraction           | `A - B`        | `__sub__`     | Same dimensions                          |
| Scalar multiplication | `A * c`, `c * A` | `__mul__`, `__rmul__` | `c` is a number                 |
| Matrix multiplication | `A @ B`        | `__matmul__`  | `A`'s columns == `B`'s rows              |
| Power                 | `A ** k`       | `__pow__`     | `A` is square, `k` is a non-negative int |
| Identity              | `A.identity()` | `identity`    | intended for square matrices             |

⚠️ Several operations currently **return** an exception object instead of raising it. See [Known limitations](#known-limitations-and-gotchas).

#### Addition and subtraction

Entry-wise: `(A + B)[i][j] = A[i][j] + B[i][j]`.

```python
>>> A = Matrix(2, 2, [[1, 2], [3, 4]])
>>> B = Matrix(2, 2, [[5, 6], [7, 8]])
>>> print(A + B)
[  6   8 ]
[ 10  12 ]
```

Implementation: `zip` pairs up matching rows, then an inner `zip` pairs up matching entries within each row.

#### Scalar multiplication

Every entry is multiplied by the scalar, which is converted to a `Fraction`. `__rmul__` makes it commutative, so `3 * A` and `A * 3` both work.

```python
>>> print(3 * A)
[ 3   6 ]
[ 9  12 ]
```

#### Matrix multiplication (`@`)

Python reserves the `@` operator for matrix multiplication (PEP 465), so `*` stays free for scalar multiplication. Entry `(i, j)` of the product is the dot product of row `i` of `A` with column `j` of `B`:

```
(A @ B)[i][j] = Σₖ A[i][k] · B[k][j]
```

An `m × n` matrix times an `n × p` matrix gives an `m × p` matrix. The inner dimensions (`n`) must match.

```python
>>> A = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
>>> B = Matrix(3, 1, [[1], [0], [2]])
>>> print(A @ B)
[  7 ]
[ 16 ]
```

Implementation: the classic triple loop (`i` over rows of `A`, `j` over columns of `B`, `k` over the shared dimension). This runs in O(m·n·p) time, which is fine for calculator-sized matrices.

#### Powers (`**`)

`A ** k` multiplies `A` by itself `k` times. Only square matrices can be raised to a power, because `A @ A` needs `A`'s columns to equal its rows.

```python
>>> A = Matrix(2, 2, [[1, -2], [3, 1]])
>>> print(A ** 0)       # identity by definition
[ 1  0 ]
[ 0  1 ]
>>> print(A ** 3)
[ -17    6 ]
[  -9  -17 ]
```

How it works:

```python
res_matrix = self  # res_matrix now *refers to* self (no copy)
for _ in range(exponent - 1):  # k - 1 multiplications give Aᵏ
    res_matrix @= self  # rebinds res_matrix to a NEW Matrix
return res_matrix
```

It starts from A¹, so it needs `k - 1` multiplications, not `k`. For why `@=` doesn't change `self`, see [Design note 3](#3-how--works-and-why-we-dont-define-__imatmul__).

#### Identity (`identity()`)

Returns an identity matrix the same size as `self`: 1s on the main diagonal (where `row == col`), 0s everywhere else. `A @ A.identity() == A` for any square `A`.

---

## Row reduction

Both methods copy the entries into `Fraction`s first, so `self` is never changed and every result is exact (`-1/2`, not `-0.5000000001`).

### Elementary row operations

REF and RREF are built from three **elementary row operations**. None of them change the solution set of the linear system a matrix represents. Each one is a private helper that takes row lists and returns new row lists:

| Operation   | Notation            | Helper            |
| ----------- | ------------------- | ----------------- |
| **Swap**    | Rᵢ ↔ Rⱼ             | `_interchange`    |
| **Scale**   | Rᵢ → c·Rᵢ (c ≠ 0)   | `_scale`          |
| **Replace** | Rᵢ → Rᵢ + c·Rⱼ      | `_replacement`    |

### `ref()`: row echelon form (Gaussian elimination)

A matrix is in **row echelon form** when:

1. Any all-zero rows are at the bottom.
2. The **pivot** (first non-zero entry) of each non-zero row is strictly to the right of the pivot in the row above.
3. All entries **below** each pivot are zero.

```
[ 2  1  -1 ]        [ 2   1    -1 ]
[ 4  5  -3 ]  REF→  [ 0   3    -1 ]
[ 6  3   0 ]        [ 0   0     3 ]
```

REF is **not unique**: different sequences of row operations give different (equally valid) REFs. Tests should check the three properties above, not one exact output.

**Algorithm:**

```
pivot_row = 0
for each column, left to right:
    find the first row at or below pivot_row with a non-zero entry in this column
    if there is none: move to the next column; pivot_row stays the same
    swap that row up into pivot_row (only if it isn't already there)
    eliminate every entry below the pivot with a replacement
    pivot_row += 1
```

**Why `pivot_row` is tracked separately from the column.** An earlier version looked for the pivot at the diagonal entry `[m][m]`, which ties the row index to the column index. When a column has no pivot (for example, a zero column), that version skipped a row as well as a column, and the pivots drifted off the staircase. `[[0, 1], [0, 2]]` came back unchanged instead of `[[0, 1], [0, 0]]`. Moving only the column forward when there's no pivot fixes this for every shape: zero columns, zero rows, and non-square matrices.

### `rref()`: reduced row echelon form (Gauss–Jordan elimination)

A matrix is in **reduced row echelon form** when it is in REF and, additionally:

4. Every pivot is exactly **1** (a "leading 1").
5. Each pivot is the **only non-zero entry in its column**.

Unlike REF, **the RREF of a matrix is unique**, so tests can compare against one exact answer.

**Algorithm:** start from `self.ref()`, then for each row from the **bottom up**:

1. Find its pivot: the leftmost non-zero entry. Skip zero rows.
2. Eliminate every entry **above** the pivot, using a replacement with the unscaled pivot row.
3. Scale the row so the pivot becomes 1.

Working bottom-up means each elimination only touches rows that haven't been finalized yet.

**What RREF unlocks:**

- **Rank:** the number of pivots.
- **Inverse:** row-reduce `[A | I]`. If the left half becomes `I`, the right half is `A⁻¹` (needs augmented matrices).
- **Solving `Ax = b`:** row-reduce `[A | b]` (needs augmented matrices).
- **Negative powers:** `A ** -k` = `(A⁻¹) ** k`.

---

## Matrix properties

These methods answer "theory" questions by looking at where the pivots of the RREF are. For an `m × n` matrix `A`:

| Method                  | True when…                       | Equivalent statements                              | Status |
| ----------------------- | -------------------------------- | -------------------------------------------------- | ------ |
| `spans_Rn()`            | there's a pivot in every **row**    | the columns span Rᵐ; `Ax = b` is consistent for every `b`; `x ↦ Ax` is onto | ✅ |
| `linear_independence()` | there's a pivot in every **column** | `Ax = 0` has only the trivial solution; `x ↦ Ax` is one-to-one | 🚧 wrong answers |
| `is_invertible()`       | `A` is square and `rref(A) == I`  | pivot in every row and column; `det(A) ≠ 0`        | ⚠️ non-square bug |

`spans_Rn()` takes the name from the usual textbook wording, but for an `m × n` matrix the space spanned is Rᵐ, where `m` is the number of **rows**.

**Design decision: compute facts in code, explain them with an LLM (later).** Every one of these properties follows exactly from the pivot positions, which the code already finds with exact arithmetic. That makes them cheap, deterministic and unit-testable. An LLM reading a matrix can miscount pivots without any sign of it, so it should only *explain* facts the code has already computed, never decide them. See [Roadmap](#roadmap).

---

## Design notes (read this before changing `Matrix`)

### 1. Operations return new objects (value semantics)

Every operator builds a fresh `_data` list and returns a new `Matrix`. Operands are never mutated. This makes the class behave like a mathematical value, similar to `int` or `Fraction`:

```python
C = A + B  # A and B are unchanged; C is a brand-new object
```

Please keep it this way. It prevents a whole category of bugs where changing one variable silently changes another.

### 2. Python assignment copies references, not objects

In Python, `x = y` **never** copies. It makes the name `x` point at the same object as `y`:

```python
B = A
B is A  # True: one object, two names
```

So if you ever mutate `B._data[0][0] = 99`, `A` changes too. When you need an independent copy, copy each row:

```python
copy = [row[:] for row in A._data]  # copies each row list
```

Note that `A._data[:]` alone is **not** enough. It copies the outer list, but the inner row lists are still shared. (`__pow__` with `k == 1` currently does exactly this; see [Known limitations](#known-limitations-and-gotchas).)

### 3. How `@=` works, and why we don't define `__imatmul__`

When Python sees `x @= y`:

1. If `x` has an `__imatmul__` method (in-place matrix multiply), Python calls it. By convention that method **mutates `x`** and returns it.
2. Otherwise Python falls back to `x = x @ y`, which calls `__matmul__` and **rebinds `x` to a new object**.

`Matrix` intentionally does **not** define `__imatmul__`, so `@=` always goes through the fallback. That is why `__pow__` is safe: `res_matrix` starts as a reference to `self`, but the first `@=` rebinds it to a new `Matrix`, and `self` is never touched.

If someone added an in-place `__imatmul__`, `__pow__` would start corrupting the original matrix. It would also produce wrong answers, because later iterations would multiply by the already-modified matrix. The same reasoning applies to `__iadd__`, `__isub__` and `__imul__`: **don't add in-place operators** unless the whole team agrees to change the design.

### 4. Row reduction works on a copy

`ref()` builds its working copy with `[[Fraction(entry) for entry in row] for row in self._data]`, which creates new row lists, and `rref()` starts from the new matrix `ref()` returns. Calling `A.rref()` never changes `A`. Keep it that way for any future method that uses row operations.

### 5. Why `Fraction`?

Row reduction divides constantly. With floats, a value that should be `0` can come out as `5.55e-17`, which breaks every "is this entry zero?" pivot check and can produce fake pivots and wrong ranks. With ints, `/` silently turns results into floats. `fractions.Fraction` does exact rational arithmetic:

```python
>>> from fractions import Fraction
>>> Fraction(1, 3) + Fraction(1, 3) + Fraction(1, 3)
Fraction(1, 1)
```

`str(Fraction(...))` already prints nicely (`1/3`, and whole numbers as `2` rather than `2/1`), so `__str__` needed no changes.

One caveat: `Fraction(0.1)` is the exact binary value of the float (`3602879701896397/36028797018963968`), not `1/10`. If users will type decimals, convert with `Fraction(str(entry))` instead.

---

## Known limitations and gotchas

These are real behaviors of the current code. Many of them make good first tickets.

| # | Issue | Example | Suggested fix |
| - | ----- | ------- | ------------- |
| 1 | **`linear_independence()` gives wrong answers.** It counts every column that has *any* non-zero entry, not just pivot columns. | `[[1, 2]]` (two vectors in R¹) → `True`; `[[1, 2], [2, 4]]` → `True` | Count pivots: `True` when the number of pivots equals the number of columns. |
| 2 | **`is_invertible()` is wrong for some non-square matrices.** `identity()` builds a non-square "identity", which a non-square RREF can match. | `[[1], [2]]` → `True` | Return `False` unless `_rows == _cols`. |
| 3 | **Errors are returned, not raised.** `__add__`, `__sub__`, `__matmul__` (type check) and `__pow__` do `return ValueError(...)` / `return TypeError(...)`, so the caller gets an exception *object* back as if it were a result. | `print(Matrix(1,1,[[1]]) + Matrix(1,2,[[1,2]]))` prints the message instead of failing | Change `return` to `raise`. For a non-`Matrix` operand, `return NotImplemented` lets Python raise the standard `TypeError`. |
| 4 | **Bare `raise ValueError`** in `__matmul__` and `__pow__` gives no message. | n/a | Include the sizes, e.g. `"Cannot multiply 2x3 by 2x3."` |
| 5 | **`__eq__` crashes on non-matrices.** `A == 5` raises `AttributeError` instead of returning `False`. | `Matrix(1,1,[[1]]) == 1` | Start with `if not isinstance(other, Matrix): return NotImplemented`. |
| 6 | **`A ** 1` shares row lists with `A`.** It copies with `self._data[:]`, which is a shallow copy (see [Design note 2](#2-python-assignment-copies-references-not-objects)). | Mutating `(A ** 1)._data[0][0]` changes `A` | Use `[row[:] for row in self._data]`. |
| 7 | **Mixed number types.** Entries are ints/floats until `ref()`/`rref()` or scalar multiplication, which return `Fraction`s. | `A._data` vs `A.rref()._data` | Convert to `Fraction` once in the constructor. |
| 8 | **Empty `data` crashes the constructor** with `IndexError` before the size check, because of `data[0]`. | `Matrix(1, 1, [])` | Check `len(data)` before reading `data[0]`, and check that every row has the same length. |

---

## Roadmap

### 1. One source of truth for pivots

Add `pivot_columns() -> list[int]`, which reads the leftmost non-zero entry of each non-zero row of the RREF. Rebuild every property on top of it:

- `rank()` = number of pivots
- `nullity()` = columns − rank
- `spans_Rn()` = rank == rows
- `linear_independence()` = rank == columns (fixes [gotcha 1](#known-limitations-and-gotchas))
- `is_invertible()` = square and rank == rows (fixes [gotcha 2](#known-limitations-and-gotchas))

Then add `analyze() -> dict`, which returns the RREF, the pivot columns, rank, nullity and every boolean property in one result. The frontend and any explanation layer both consume it.

### 2. Augmented matrix support

An **augmented matrix** `[A | b]` glues the coefficient matrix of a linear system to its right-hand side. The system

```
 x + 2y = 5
3x + 4y = 6
```

becomes

```
[ 1  2 | 5 ]
[ 3  4 | 6 ]
```

Row-reducing an augmented matrix to RREF solves the system. The math is identical to ordinary RREF; the only differences are:

- **Remember where the divider is**, so we can print it and interpret the result.
- **Pivots should only be chosen from the coefficient columns** (left of the divider). A pivot in the right-hand column means the system is *inconsistent* (no solution), e.g. the row `[ 0  0 | 1 ]`, which says `0 = 1`.

**Proposed design** (open for discussion):

- Add an optional attribute to `Matrix`, e.g. `augment_col: int | None = None`, the index of the first column to the right of the divider. `None` means an ordinary matrix.
- Add a constructor helper, e.g. `Matrix.augment(A, B) -> Matrix`, that checks `A` and `B` have the same number of rows, glues each row of `A` to the matching row of `B`, and sets `augment_col` to `A`'s column count. Allowing `B` to have several columns covers both `[A | b]` (solving) and `[A | I]` (inverting).
- Update `__str__` to print a `|` at the divider, and `__repr__` to include `augment_col`.
- `ref()` / `rref()` keep the divider on the returned matrix, and restrict pivot search to columns `< augment_col`.
- Reject arithmetic on augmented matrices (`A + B` where either is augmented → `ValueError`), since it rarely makes mathematical sense.

**Interpreting the result:** once an augmented matrix is in RREF, we can classify the system:

| RREF shows…                                                         | Meaning                                                  |
| ------------------------------------------------------------------- | -------------------------------------------------------- |
| A row `[ 0 … 0 \| c ]` with `c ≠ 0`                                  | **No solution** (inconsistent)                           |
| A pivot in every coefficient column                                 | **Exactly one solution**; read it off the right-hand column |
| At least one coefficient column without a pivot (a *free variable*) | **Infinitely many solutions**                            |

A later `solve()` method can build on this classification.

### 3. Other planned work

- **Unit tests** for everything that exists today (see [Testing](#testing)).
- **Fixes from [Known limitations](#known-limitations-and-gotchas)**, starting with raising errors properly, since tests depend on it.
- **More operations:** transpose, determinant, inverse, trace.
- **"Show your work":** record each row operation during `ref()` / `rref()` so the calculator can display the steps.
- **Optional LLM explanations:** a separate module that takes the `analyze()` result and asks a model to explain each fact and the theorem behind it, treating the computed values as ground truth. Keep it optional, so the calculator works without an API key and tests never call the network.
- **Web API:** FastAPI and Uvicorn are already dependencies. The plan is to expose the calculator through HTTP endpoints (e.g. `POST /matrix/rref`), converting between JSON and `Matrix` at the boundary. The math in `core/` should stay independent of the web layer.

---

## Testing

Tests live in `backend/tests/` and run with **pytest**:

```bash
cd backend
uv run pytest          # run everything
uv run pytest -v       # verbose: show each test name
uv run pytest -k rref  # only tests whose name contains "rref"
```

There are **no tests yet**; this is the next thing to add. Suggested layout:

```
backend/tests/
├── conftest.py            ← shared fixtures (e.g. common example matrices)
├── test_matrix_basics.py  ← construction, __str__, __repr__, __eq__, validation
├── test_arithmetic.py     ← +, -, *, @, **, identity()
├── test_row_reduction.py  ← ref(), rref()
└── test_properties.py     ← spans_Rn(), linear_independence(), is_invertible()
```

Example tests:

```python
import pytest
from linalg_calc.core.matrix import Matrix


def test_matmul_2x2():
    A = Matrix(2, 2, [[1, 2], [3, 4]])
    B = Matrix(2, 2, [[5, 6], [7, 8]])
    assert A @ B == Matrix(2, 2, [[19, 22], [43, 50]])


def test_rref_zero_column():
    A = Matrix(2, 2, [[0, 1], [0, 2]])
    assert A.rref() == Matrix(2, 2, [[0, 1], [0, 0]])


def test_rref_does_not_mutate_original():
    A = Matrix(2, 2, [[2, 4], [1, 3]])
    A.rref()
    assert A == Matrix(2, 2, [[2, 4], [1, 3]])
```

What to test for every operation:

- **The happy path** with a small, hand-checkable example.
- **Non-square shapes** (2×3, 3×1, 1×1), not just square matrices.
- **Error cases:** wrong dimensions, wrong operand types.
- **No mutation:** the operands are unchanged afterward.
- **Row reduction specifically:** zero pivots that need a row swap, all-zero rows, all-zero columns, a pivot in the last column, singular matrices, and decimal entries.
- **Properties specifically:** wide matrices (more columns than rows), tall matrices (more rows than columns), and `[[1, 2]]`, which is dependent but spans R¹.

---

## Coding conventions

- **Formatting and linting:** run `uv run ruff format .` and `uv run ruff check .` before pushing.
- **Type hints on every function signature.** Use the string form `"Matrix"` when a method refers to its own class (the class isn't fully defined yet while its body is being read).
- **Google-style docstrings** with `Args:`, `Returns:` and `Raises:` sections, as the existing methods have. Document every exception a method can raise.
- **Never mutate operands.** Return a new `Matrix`. See [Design notes](#design-notes-read-this-before-changing-matrix).
- **Raise errors, don't return them.** Error messages should say what went wrong and include the sizes, e.g. `"Cannot add 2x3 and 3x2 matrices."`
- **Keep `core/` pure.** No printing, no I/O, no web code in the math layer.
- **Commit `uv.lock`.** It guarantees everyone runs the same dependency versions.

---

## Glossary

| Term | Meaning |
| ---- | ------- |
| **m × n matrix** | A matrix with *m* rows and *n* columns. Rows are always listed first. |
| **Square matrix** | A matrix with the same number of rows and columns. |
| **Identity matrix (I)** | A square matrix with 1s on the main diagonal and 0s elsewhere. `A @ I == A`. |
| **Scalar** | A single number, as opposed to a matrix. |
| **Dunder method** | A "double underscore" method like `__add__` that Python calls for built-in syntax like `+`. |
| **Pivot / leading entry** | The first non-zero entry in a row of an echelon form. |
| **Pivot column** | A column that contains a pivot. |
| **REF** | Row echelon form: a staircase shape with zeros below each pivot. Not unique. |
| **RREF** | Reduced row echelon form: REF, plus every pivot is 1 and is the only non-zero entry in its column. Unique. |
| **Elementary row operations** | Swap, scale and replace: the three row moves allowed during row reduction. |
| **Gaussian elimination** | The algorithm that produces REF. |
| **Gauss–Jordan elimination** | The algorithm that produces RREF. |
| **Rank** | The number of pivots in the REF/RREF. |
| **Nullity** | The number of columns without a pivot (free variables). Rank + nullity = number of columns. |
| **Span** | All linear combinations of a set of vectors. The columns of `A` span Rᵐ when there's a pivot in every row. |
| **Linearly independent** | No vector in the set is a combination of the others. The columns of `A` are independent when there's a pivot in every column. |
| **Invertible** | A square matrix `A` with an `A⁻¹` such that `A @ A⁻¹ == I`. Equivalent to `rref(A) == I`. |
| **Augmented matrix** | A coefficient matrix and a right-hand side joined together, written `[A \| b]`. |
| **Free variable** | A variable whose column has no pivot in the RREF; it can take any value. |
| **Inconsistent system** | A system with no solution, shown by a row `[0 … 0 \| c]` with `c ≠ 0`. |
