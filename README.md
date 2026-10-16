# Linear Algebra Calculator

A linear algebra calculator written in Python. At its core is a small, dependency-free `Matrix` class that supports the standard matrix operations using ordinary Python operators (`+`, `-`, `*`, `@`, `**`). It can also row-reduce matrices to REF and RREF using exact fraction arithmetic, and answer theory questions from the pivots: rank, nullity, "do the columns span Rᵐ?", "are the columns linearly independent?" and "is this matrix invertible?". Next up: unit tests, then augmented matrices for solving systems of linear equations, and eventually a web API (FastAPI).

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
| Equality (`==`)                        | ✅ Done (comparing to a non-`Matrix` still errors, see [gotcha 1](#known-limitations-and-gotchas)) |
| Addition / subtraction                 | ✅ Done (errors are raised properly)                |
| Scalar multiplication (`A * c`, `c * A`) | ✅ Done                                           |
| Matrix multiplication                  | ✅ Done (non-`Matrix` operand gives a confusing error, see [gotcha 1](#known-limitations-and-gotchas)) |
| Integer powers (`A ** k`, k ≥ 0)       | ✅ Done                                             |
| Identity matrix                        | ✅ Done                                             |
| Exact arithmetic with `Fraction`       | ✅ Done (inside `ref()` / `rref()`)                 |
| REF (row echelon form)                 | ✅ Done                                             |
| RREF (reduced row echelon form)        | ✅ Done                                             |
| `pivot_rows()`, `pivot_cols()`, `rank()` | ✅ Done                                           |
| `dim_col_space()`, `dim_null_space()`  | ✅ Done                                             |
| `spans_rn()`, `linear_independence()`  | ✅ Done                                             |
| `is_invertible()`                      | ✅ Done                                             |
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

`+` and `-` raise `TypeError` for a non-`Matrix` operand and `ValueError` for mismatched sizes. `@` and `**` raise `ValueError` with a message when the dimensions don't fit, and `**` also raises it for a negative exponent.

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

### Pivot counting helpers

| Method             | Returns                                                  | Example: `[[1, 2], [2, 4]]` |
| ------------------ | -------------------------------------------------------- | --------------------------- |
| `pivot_rows()`     | number of non-zero rows in the RREF (= number of pivots) | `1` |
| `pivot_cols()`     | number of pivot columns, counted from the RREF           | `1` |
| `rank()`           | `pivot_cols()`                                           | `1` |
| `dim_col_space()`  | `pivot_cols()` (dim Col A = rank)                        | `1` |
| `dim_null_space()` | `cols - pivot_cols()` (dim Nul A = nullity)              | `1` |

Every RREF row has at most one pivot and every pivot sits in its own column, so the number of pivot rows always equals the number of pivot columns: both are the rank. That's why `pivot_cols()` can count non-zero RREF rows, and why rank + nullity always equals the number of columns.

### Boolean properties

| Method                  | True when…                       | Equivalent statements                              | Implemented as |
| ----------------------- | -------------------------------- | -------------------------------------------------- | -------------- |
| `spans_rn()`            | there's a pivot in every **row**    | the columns span Rᵐ; `Ax = b` is consistent for every `b`; `x ↦ Ax` is onto | `pivot_cols() == rows` |
| `linear_independence()` | there's a pivot in every **column** | `Ax = 0` has only the trivial solution; `x ↦ Ax` is one-to-one | `pivot_cols() == cols` |
| `is_invertible()`       | `A` is square and `rref(A) == I`  | pivot in every row and column; `det(A) ≠ 0`        | square and `rref() == identity()` |

How shape limits the answers:

| Matrix | `spans_rn()` | `linear_independence()` | Why |
| ------ | ------------ | ----------------------- | --- |
| `[[1, 2]]` (wide, 1×2) | `True` | `False` | 2 vectors in R¹ must be dependent |
| `[[1, 0], [0, 1], [1, 1]]` (tall, 3×2) | `False` | `True` | 2 vectors can't span R³ |
| `[[1, 2], [2, 4]]` (singular) | `False` | `False` | rank 1 < 2 |
| `[[1, 2], [3, 4]]` (invertible) | `True` | `True` | rank 2 = rows = columns |

`spans_rn()` (renamed from `spans_Rn()` to follow snake_case) takes the name from the usual textbook wording, but for an `m × n` matrix the space spanned is Rᵐ, where `m` is the number of **rows**.

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

Note that `A._data[:]` alone is **not** enough. It copies the outer list, but the inner row lists are still shared. (`__pow__` with `k == 1` used to make exactly this mistake; it now copies each row.)

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
| 1 | **`__eq__` and `__matmul__` call `NotImplemented(...)`** for a non-`Matrix` operand. `NotImplemented` is a constant, not a function, so this raises `TypeError: 'NotImplementedType' object is not callable` instead of returning `False` (for `==`) or a clear error (for `@`). | `A == 5`, `A @ 5` | `return NotImplemented` (no parentheses, no message). Python then makes `A == 5` return `False` and `A @ 5` raise the standard `TypeError`. |
| 2 | **Docstrings for `__add__` / `__sub__` say `NotImplementedError`**, but the code raises `TypeError`. `__pow__` on a non-square matrix reuses the `@` error message. | n/a | Update the `Raises:` sections and give `__pow__` its own message, e.g. `"Only square matrices can be raised to a power."` |
| 3 | **Mixed number types.** Entries are ints/floats until `ref()`/`rref()` or scalar multiplication, which return `Fraction`s. The constructor's type hint now says `list[list[Fraction]]`, but nothing converts the input. | `A._data` vs `A.rref()._data` | Convert to `Fraction` once in the constructor. |
| 4 | **Empty `data` still crashes the constructor** with `IndexError`: the `any((...))` tuple evaluates `data[0]` before `any` runs, so `not data` doesn't protect it. **Ragged rows aren't caught**, since only the first row's length is checked. | `Matrix(1, 1, [])`; `Matrix(2, 2, [[1, 2], [3]])` is accepted | Check `len(data)` first with `or` (which short-circuits), then `all(len(row) == cols for row in data)`. |
| 5 | **`identity()` on a non-square matrix** returns a non-square "identity". `is_invertible()` now guards against this, but direct calls don't. | `Matrix(2, 3, ...).identity()` | Raise `ValueError` unless the matrix is square. |

---

## Roadmap

### 1. One source of truth for pivots

**Done:** `pivot_rows()`, `pivot_cols()`, `rank()`, `dim_col_space()`, `dim_null_space()`, `spans_rn()` and `linear_independence()` all count pivots from the RREF.

Next step: each of those calls `rref()` again, and `pivot_rows()` and `pivot_cols()` compute the same number. Add `pivot_columns() -> list[int]`, which returns the *positions* of the pivots (the leftmost non-zero entry of each non-zero row of the RREF), not just a count. Then rebuild every property on top of it, so one RREF answers everything:

- `rank()` = number of pivots
- `dim_null_space()` (nullity) = columns − rank
- `spans_rn()` = rank == rows
- `linear_independence()` = rank == columns
- `is_invertible()` = square and rank == rows (cheaper than building and comparing an identity matrix)

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
- **Fixes from [Known limitations](#known-limitations-and-gotchas)**, starting with the `NotImplemented` calls and the constructor checks.
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
└── test_properties.py     ← pivot helpers, rank(), spans_rn(), linear_independence(), is_invertible()
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
