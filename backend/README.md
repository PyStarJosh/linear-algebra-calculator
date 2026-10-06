# linalg_calc (backend)

The Python backend for the Linear Algebra Calculator. It has two jobs:

1. **Compute.** An exact, dependency-free `Matrix` class does the arithmetic, row reduction (REF / RREF) and matrix-property checks.
2. **Explain** *(planned).* An optional LLM layer turns the computed results into step-by-step explanations a student can follow.

The rule that ties them together: **the code decides every fact, and the LLM only explains facts the code has already computed.**

> For the full math background, design notes, known bugs and contributor conventions, see the [root README](../README.md).

---

## Quick start

```bash
cd backend
uv sync                                   # create .venv and install dependencies
uv run python src/linalg_calc/main.py     # run the sandbox script
uv run pytest                             # run tests (none yet)
uv run ruff check . && uv run ruff format .
```

```python
>>> from linalg_calc.core.matrix import Matrix
>>> A = Matrix(rows=3, cols=4, data=[[1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 2, 1]])
>>> print(A.rref())
[ 1  0  0     0 ]
[ 0  1  0  -1/2 ]
[ 0  0  1   1/2 ]
>>> A.spans_Rn()
True
```

---

## Layout

```
backend/
├── pyproject.toml            ← metadata, dependencies, entry point
├── uv.lock
├── src/linalg_calc/
│   ├── __init__.py           ← package entry point: main()
│   ├── main.py               ← sandbox script
│   └── core/
│       └── matrix.py         ← Matrix class: all the math lives here
└── tests/
    └── conftest.py           ← shared fixtures (empty for now)
```

---

## Current implementation

All of this lives in `src/linalg_calc/core/matrix.py`. Every method returns a new `Matrix` (or a `bool`) and never changes `self`.

### Operations

| Feature                    | Usage                       | Status |
| -------------------------- | --------------------------- | ------ |
| Construction and printing  | `Matrix(rows, cols, data)`, `print(A)`, `repr(A)` | ✅ |
| Equality                   | `A == B`                    | ✅ |
| Addition / subtraction     | `A + B`, `A - B`            | ✅ (errors are returned instead of raised) |
| Scalar multiplication      | `A * c`, `c * A`            | ✅ |
| Matrix multiplication      | `A @ B`                     | ✅ |
| Integer powers             | `A ** k` (k ≥ 0, square)    | ✅ |
| Identity                   | `A.identity()`              | ✅ |

### Row reduction

| Method   | Algorithm                  | Notes |
| -------- | -------------------------- | ----- |
| `ref()`  | Gaussian elimination       | ✅ Handles zero columns, zero rows and non-square shapes by tracking `pivot_row` separately from the column. |
| `rref()` | Gauss–Jordan, bottom-up    | ✅ Starts from `ref()`, clears the entries above each pivot, then scales the pivot to 1. |

Both convert entries to `fractions.Fraction`, so results are exact (`-1/2`, never `-0.4999999`). They're built on three private row-operation helpers: `_interchange` (swap), `_scale` and `_replacement`.

### Matrix properties

All three are read off the pivot positions of the RREF, for an `m × n` matrix:

| Method                  | True when…                    | Status |
| ----------------------- | ----------------------------- | ------ |
| `spans_Rn()`            | there's a pivot in every row (the columns span Rᵐ)      | ✅ |
| `linear_independence()` | there's a pivot in every column                        | 🚧 counts any non-zero entry, not just pivots, so `[[1, 2]]` → `True` (wrong) |
| `is_invertible()`       | `A` is square and `rref(A) == I` | ⚠️ returns `True` for some non-square matrices, e.g. `[[1], [2]]` |

### Next step in `core/`: one source of truth

Before the LLM layer, centralize the pivot logic:

- `pivot_columns() -> list[int]`: the leftmost non-zero entry of each non-zero RREF row.
- `rank`, `nullity`, `spans_Rn`, `linear_independence` and `is_invertible` each become a one-line check on top of it. This fixes both bugs above.
- `analyze() -> dict`: computes the RREF **once** and returns every fact in a single, JSON-serializable result. The web API and the LLM layer both consume it.

```python
>>> Matrix(2, 3, [[1, 2, 3], [0, 0, 1]]).analyze()
{
    "shape": [2, 3],
    "input": [["1", "2", "3"], ["0", "0", "1"]],
    "rref": [["1", "2", "0"], ["0", "0", "1"]],
    "pivot_columns": [0, 2],
    "rank": 2,
    "nullity": 1,
    "spans_Rm": True,               # pivot in every row
    "linearly_independent": False,  # column 1 has no pivot
    "invertible": None,             # not square, so the question doesn't apply
}
```

Entries are serialized as strings (`"1/2"`), so exact fractions survive JSON. `None` means "not applicable", which keeps it distinct from `False`.

---

## Future scope: LLM explanation layer

### Goal

After a computation, a user can ask *why*. For example: "Why don't these columns span R³?" or "Explain each row operation." The LLM writes the explanation in plain language and cites the relevant theorem. It never computes or decides the answer itself.

### Why the LLM doesn't do the math

- **Correctness:** LLMs can miscount pivots or confuse rows with columns, and they give no sign when they do. A calculator can't present wrong yes/no answers.
- **Testability:** booleans from `analyze()` can be unit-tested. LLM prose can't, in any reliable way.
- **Cost and availability:** computing is instant, free and works offline. The calculator must work fully with no API key.

### Architecture

```
user input
   │
   ▼
Matrix(...) ──► analyze() ──► facts (dict)          ← core/: deterministic, tested
                                  │
                                  ├──► response.facts          (always returned)
                                  │
                                  ▼  (only if explanations are enabled)
                            Explainer.explain(facts, question)
                                  │
                                  ├─ prompts.build(facts, question)
                                  ├─ LLMClient.complete(prompt)   ← the only network call
                                  └─ validate(output, facts)
                                  │
                                  ▼
                            response.explanation       (None if disabled or failed)
```

Planned modules:

```
src/linalg_calc/
├── core/matrix.py            ← unchanged: knows nothing about LLMs
└── explain/
    ├── __init__.py           ← public entry point: explain(facts, question=None)
    ├── client.py             ← LLMClient protocol + provider implementations
    ├── prompts.py            ← system prompt and prompt builder
    └── validate.py           ← checks the output against the facts
```

### Design rules

1. **Dependency direction:** `explain/` imports from `core/`, never the other way round. `core/` stays pure math.
2. **Provider-agnostic client.** The explainer depends on a small protocol, not on a specific SDK:

   ```python
   from typing import Protocol


   class LLMClient(Protocol):
       def complete(self, system: str, user: str) -> str: ...
   ```

   `OpenAIClient` is the first implementation (it wraps the official `openai` SDK). Swapping providers, or using a fake in tests, means writing one small class.
3. **Facts are ground truth.** The system prompt states that every value in the facts is correct and must not be recomputed or contradicted. The prompt includes the facts as JSON, plus the question if there is one.
4. **Structured output.** The model is asked for JSON with one explanation per fact, e.g. `{"spans_Rm": "...", "linearly_independent": "..."}`, so the frontend can show each explanation next to its result.
5. **Validation.** `validate.py` rejects output that leaves out a requested fact or contradicts a boolean (for example, saying "independent" when the fact is `False`). On failure, retry once, then return no explanation. Never return a wrong one.
6. **Fail soft.** A missing key, a network error, a timeout or a validation failure all lead to `explanation = None`. The computed facts are always returned.
7. **Opt-in.** Explanations only run when they're configured and requested.

### Configuration (planned)

| Variable                  | Purpose                                            |
| ------------------------- | -------------------------------------------------- |
| `LINALG_EXPLAIN_ENABLED`  | `true` to turn explanations on (default: off)      |
| `LINALG_LLM_PROVIDER`     | Which `LLMClient` implementation to use (default: `openai`) |
| `LINALG_LLM_MODEL`        | Model name passed to the provider                  |
| `OPENAI_API_KEY`          | Read by the OpenAI SDK; **never commit it**         |

Keep keys in a local `.env` file (add `.env` to `.gitignore`), or in your shell environment. When the time comes, add the SDK as an optional dependency group so the core calculator stays dependency-free:

```bash
uv add --optional llm openai
uv sync --extra llm
```

### Testing the LLM layer

- **Unit tests never call the network.** They inject a `FakeLLMClient` that returns canned JSON.
- **Prompt tests** check that `prompts.build()` includes every fact and the ground-truth instruction.
- **Validation tests** feed in contradictory or incomplete output and check that it's rejected.
- **Failure tests** check that a client which raises still returns the facts, with `explanation=None`.
- **Live tests** (optional) are marked `@pytest.mark.llm` and skipped unless an API key is set.

### Possible follow-up: "show your work"

If `ref()` and `rref()` record each row operation (e.g. `R2 → R2 − 3·R1`), the steps can go into `facts` too. The LLM can then explain *why* each step was chosen, again without computing anything itself.

---

## Roadmap (backend)

1. **Unit tests** for everything in `core/` (see the root README for suggested cases).
2. **Fix `core/` bugs:** raise errors instead of returning them, fix `linear_independence()` and `is_invertible()`, and convert entries to `Fraction` in the constructor.
3. **Add `pivot_columns()`, `rank`, `nullity` and `analyze()`.**
4. **Augmented matrices** `[A | b]` and `solve()`.
5. **FastAPI endpoints**, e.g. `POST /matrix/analyze` → `{facts, explanation}`.
6. **`explain/` package** with the OpenAI client, validation and tests.
