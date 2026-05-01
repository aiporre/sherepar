from __future__ import annotations

from typing import Tuple

import numpy as np


def extract_roi_patch(
        vertices: np.ndarray,
        center: np.ndarray,
        radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract vertices within *radius* of *center* using a KD-tree query.

    Uses scipy.spatial.cKDTree (O(n log n) construction, O(log n) per query)
    for efficient spatial indexing.

    Parameters
    ----------
    vertices:
        All mesh vertices, shape (N, 3).
    center:
        Query point, shape (3,).
    radius:
        Search radius.

    Returns
    -------
    patch_vertices : np.ndarray, shape (K, 3)
        Positions of vertices in the patch.
    patch_indices : np.ndarray, shape (K,)
        0-based indices into *vertices*.
    """
    from scipy.spatial import cKDTree  # pylint: disable=import-outside-toplevel

    tree = cKDTree(vertices)
    indices = np.array(tree.query_ball_point(center, radius), dtype=np.intp)
    if len(indices) == 0:
        # Fallback: nearest single vertex
        print("Fallback no nearest in radius ", radius)
        _, idx = tree.query(center)
        indices = np.array([idx], dtype=np.intp)
    return vertices[indices], indices
