import numpy as np


def squared_error(heights, goal_heights):
    """
    Squared distance from the ``heights`` to ``goal heights``.

    :param heights: Actual heights of  shape ``[..., n_height_vars]``.
    :param goal_heights: Goal heights of shape ``[..., n_height_vars]``.
    :return: Mean squared error between actual and goal heights, summed over time.
    """
    # Sums squared error over height variables.
    distance = np.sum(
        np.square(heights - goal_heights),
        axis=-1
    )
    return distance


def mean_cumulative_squared_error(heights, goal_heights, weights=None):
    """
    Mean (over episodes) cumulative (over time) squared distance from ``heights`` to ``goal_heights``.

    :param heights: Actual heights of shape ``[n_episodes, time, n_height_vars]``.
    :param goal_heights: Goal heights of shape ``[n_episodes, time, n_height_vars]``.
    :param weights: Weights to multiply squared error over the time dimension. Defaults to all ones.
    :return: Mean cumulative squared error between actual and goal machine heights.
    """
    # Initializes weights to all ones if not specified.
    if weights is None:
        weights = np.ones(heights.shape[0])

    # Computes mean cumulative squared error weighted by ``weights.``
    distance = np.mean(
        np.sum(
            weights * squared_error(heights, goal_heights),
            axis=-1
        ),
        axis=0
    )
    return distance


def absolute_error(heights, goal_heights):
    """
    Absolute error from the goal heights.

    :param heights: Actual heights. Should have shape ``[..., n_height_vars]``.
    :param goal_heights: Goal heights. Should have shape ``[..., n_height_vars]``.
    :return: Absolute error between actual and goal heights, summed over time.
    """
    # Sums absolute error over height variables.
    distance = np.sum(
        np.abs(heights - goal_heights),
        axis=-1
    )
    return distance


def mean_cumulative_absolute_error(heights, goal_heights, weights=None):
    """
    Mean (over episodes) cumulative (over time) absolute distance from ``heights`` to ``goal_heights``.

    :param heights: Actual heights of shape ``[n_episodes, time, n_height_vars]``.
    :param goal_heights: Goal heights of shape ``[n_episodes, time, n_height_vars]``.
    :param weights: Weights to multiply squared error over the time dimension. Defaults to all ones.
    :return: Mean cumulative absolute error between actual and goal machine heights.
    """
    # Initializes weights to all ones if not specified.
    if weights is None:
        weights = np.ones(heights.shape[0])

    # Computes mean cumulative squared error weighted by ``weights.``
    distance = np.mean(
        np.sum(
            weights * absolute_error(heights, goal_heights),
            axis=-1
        ),
        axis=0
    )
    return distance
