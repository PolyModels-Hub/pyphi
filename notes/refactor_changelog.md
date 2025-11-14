# Refactor Changelog

This log tracks the gradual migration of functionality from the legacy
`pyphi.py` module into the new modular package.

Entries should include:

- Date of the change
- Functions duplicated or moved
- Associated tests that cover the change
- Any follow-up items (e.g., consumers still using the legacy copy)

## Entries

- 2025-01-15: Duplicated `mean`, `std`, and `meancenterscale` into `src/pyphi/utils.py`; added comprehensive tests in `tests/test_utils.py`; legacy implementations remain in `pyphi.py`.
