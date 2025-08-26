import numpy as np
from scipy.stats import wasserstein_distance


def persistence_wasserstein(diagram1, diagram2, dimension=1):
    d1 = (
        diagram1[dimension] if len(diagram1) > dimension else np.array([]).reshape(0, 2)
    )
    d2 = (
        diagram2[dimension] if len(diagram2) > dimension else np.array([]).reshape(0, 2)
    )

    # Remove infinite persistence features for Wasserstein computation
    d1_finite = d1[d1[:, 1] != np.inf]
    d2_finite = d2[d2[:, 1] != np.inf

    if len(d1_finite) == 0 and len(d2_finite) == 0:
        return 0.0

    # Use simplified Wasserstein distance on persistence values
    persistence1 = (
        d1_finite[:, 1] - d1_finite[:, 0] if len(d1_finite) > 0 else np.array([])
    )
    persistence2 = (
        d2_finite[:, 1] - d2_finite[:, 0] if len(d2_finite) > 0 else np.array([])
    )

    # Compute Wasserstein distance
    return wasserstein_distance(persistence1, persistence2)
