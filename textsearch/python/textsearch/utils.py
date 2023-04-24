import numpy as np
from _fasttextsearch import row_ids_to_row_splits as _row_ids_to_row_splits


def row_ids_to_row_splits(row_ids: np.ndarray) -> np.ndarray:
    """Convert row ids to row splits.

    Args:
      row_ids:
        A 1-D array of dtype np.uint32 containing row ids.
    Returns:
      Return a 1-D array of dtype np.uint32 containing row splits.
    """
    assert row_ids.ndim == 1, row_ids.ndim
    assert row_ids.dtype == np.uint32, row_ids.dtype

    row_ids = np.ascontiguousarray(row_ids)
    num_rows = row_ids[-1] + 1
    row_splits = np.empty(num_rows + 1, dtype=np.uint32)

    _row_ids_to_row_splits(row_ids, row_splits)
    return row_splits
