import numpy as np
from typing import List, Tuple, Optional

def _largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of the largest 4-connected component in `mask`.

    Args:
        mask: Boolean numpy array where True indicates a component.

    Returns:
        Boolean numpy array with the largest component marked as True.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_coords = []

    for r in range(h):
        for c in range(w):
            if mask[r, c] and not visited[r, c]:
                stack = [(r, c)]
                coords = []
                visited[r, c] = True

                while stack:
                    y, x = stack.pop()
                    coords.append((y, x))

                    for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

                if len(coords) > len(best_coords):
                    best_coords = coords

    out = np.zeros_like(mask, dtype=bool)
    if best_coords:
        ys, xs = zip(*best_coords)
        out[np.array(ys), np.array(xs)] = True

    return out

def solve_arc_agi_1(
    input_grid: List[List[int]],
    choose: str = "largest",  # "largest" (by area) or "all"
    align: str = "origin"     # "origin" = keep original location; "topleft" = move bbox to (0,0)
) -> List[List[int]]:
    """
    Copy a block to a new grid of the same size, filling the rest with 0s.

    Args:
        input_grid: 2D list of integers representing the input grid.
        choose: Strategy for selecting pixels to copy:
            - "largest": Copy only the largest 4-connected non-zero component.
            - "all": Copy all non-zero pixels.
        align: Strategy for aligning the copied block:
            - "origin": Keep pixels at their original coordinates.
            - "topleft": Translate the selected component's bounding box so its
                top-left corner is at (0,0), still within a grid of the same size.

    Returns:
        2D list of integers representing the solved grid.
    """
    arr = np.array(input_grid, dtype=int)
    rows, cols = arr.shape

    if not np.any(arr):
        return np.zeros((rows, cols), dtype=int).tolist()

    mask_all = arr != 0

    if choose == "largest":
        mask = _largest_component(mask_all)
    else:
        mask = mask_all

    # Bounding box of chosen pixels
    ys, xs = np.where(mask)
    min_r, max_r = ys.min(), ys.max()
    min_c, max_c = xs.min(), xs.max()

    out = np.zeros_like(arr)

    if align == "origin":
        # Copy only chosen pixels at their original positions
        out[mask] = arr[mask]
    else:  # align == "topleft"
        # Translate bbox so its top-left moves to (0,0)
        dy, dx = -min_r, -min_c
        ys_new = ys + dy
        xs_new = xs + dx
        out[ys_new, xs_new] = arr[ys, xs]

    return out.tolist()

# Example usage
if __name__ == '__main__':
    example_input = [
        [0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0],
        [0, 8, 8, 8, 0],
        [0, 8, 8, 8, 0],
        [0, 0, 0, 0, 0]
    ]

    print("Input Grid:")
    for r in example_input:
        print(r)

    solved = solve_arc_agi_1(example_input, choose="largest", align="origin")
    print("\nSolved Grid:")
    for r in solved:
        print(r)
