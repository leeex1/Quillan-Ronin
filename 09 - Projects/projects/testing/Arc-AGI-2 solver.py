import numpy as np
from typing import List
from collections import Counter


def solve_arc_agi_2(input_grid: List[List[int]]) -> List[List[int]]:
    """
    Solves the ARC-AGI-2 task by placing a single pixel at the centroid
    of the non-zero cells, with robust tie-breaking for even-sized cases,
    and choosing the dominant non-zero color.
    """
    arr = np.array(input_grid, dtype=int)
    rows, cols = arr.shape


    # All foreground cells
    nz = np.argwhere(arr != 0)
    if nz.size == 0:
        return np.zeros((rows, cols), dtype=int).tolist()


    # Choose color = most frequent non-zero value
    vals = arr[nz[:, 0], nz[:, 1]].tolist()
    color = Counter(vals).most_common(1)[0][0]


    # Bounding box (keep placement anchored to the object region)
    min_row, min_col = nz.min(axis=0)
    max_row, max_col = nz.max(axis=0)


    # Geometric centroid of occupied cells (float y, x)
    cy, cx = nz.mean(axis=0)


    # Candidate integer centers near the centroid (handles .5 ties)
    candidates = set()
    for ry in (np.floor(cy), np.ceil(cy)):
        for cx_ in (np.floor(cx), np.ceil(cx)):
            y = int(np.clip(ry, 0, rows - 1))
            x = int(np.clip(cx_, 0, cols - 1))
            candidates.add((y, x))


    # Prefer candidates that lie inside the bounding box
    candidates = [(y, x) for (y, x) in candidates
                  if (min_row <= y <= max_row and min_col <= x <= max_col)]


    if not candidates:
        # Fallback: clamp rounded centroid; avoid bankers’ rounding surprises by casting via floor of (cy+0.5)
        y = int(np.clip(np.floor(cy + 0.5), 0, rows - 1))
        x = int(np.clip(np.floor(cx + 0.5), 0, cols - 1))
        center = (y, x)
    else:
        # Choose the candidate closest to (cy, cx); tie-break to upper-left (smaller y, then x)
        center = min(candidates, key=lambda p: ((p[0] - cy) ** 2 + (p[1] - cx) ** 2, p[0], p[1]))


    out = np.zeros((rows, cols), dtype=int)
    out[center] = color
    return out.tolist()


# Example usage
if __name__ == '__main__':
    example_input = [
        [0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 8, 0],
        [0, 8, 0, 0, 8, 0],
        [0, 8, 8, 8, 8, 0],
        [0, 0, 0, 0, 0, 0]
    ]
    solved_grid = solve_arc_agi_2(example_input)
