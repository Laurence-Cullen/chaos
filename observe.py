import numpy as np


def observe(data, std):
    """
    Measure 'perfect' data to produce observed values by adding normally distributed errors to each element of data.

    Args:
        data: The 'perfect' data to synthetically observe.
        std: The standard deviation of the normal distribution to generate errors with.

    Returns:
        ndarray
    """
    return np.random.normal(data, scale=std, size=data.shape)
