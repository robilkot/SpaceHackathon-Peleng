from dataclasses import dataclass

import numpy as np


@dataclass
class Line:
    point: np.ndarray
    vector: np.ndarray


def find_point(
    line1: Line,
    line2: Line
) -> np.ndarray:
    perp = np.cross(line1.vector, line2.vector)
    a = np.hstack(
        (
            line1.vector.reshape((-1, 1)),
            line2.vector.reshape((-1, 1)),
            perp.reshape((-1, 1))
        )
    )
    b = line1.point - line2.point
    *t, _ = np.linalg.tensorsolve(a, b)
    h1 = line1.vector * t[0] + line1.point
    h2 = line2.vector * t[1] + line2.point
    return np.mean((h1, h2), axis=(0,))


if __name__ == '__main__':
    line1 = Line(np.asarray((0., 0., 0.)), np.asarray((1., 0., 0.)))
    line2 = Line(np.asarray((0., 0., 0.)), np.asarray((0., 1., 0.)))
    print(find_point(line1, line2))
