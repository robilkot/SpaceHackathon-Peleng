import numpy as np


def find_point(
    point: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    perp = np.cross(*vector)
    
