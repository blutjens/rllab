import numpy as np

from rllab.envs.envs import constants


def shift_and_scale(goal):
    """
    Shifts and scales goal into valid height range.

    :param goal: Floating point number in ``[-1, 1]``.
    :return: Number translated into valid goal height range.
    """
    goal = (goal + 1) / 2
    return goal * (constants.goal_max - constants.goal_min) + constants.goal_min


class SineTask:
    """
    A callable class that implements a sine wave to use as a goal height trajectory. Computes values in ``[-1, 1]`` and
    then shifts and scales them into ``[constants.goal_min, constants.goal_max]``.
    """

    def __init__(self, steps=500, periods=1., offset=0.):
        """
        Constructor for ``SineTask`` object.

        :param steps: Length of sine wave.
        :param periods: Number of periods of sine wave.
        :param offset: Sine wave evaluation initial condition. In ``[0, 2 * np.pi]``.
        """
        self.steps = steps
        self.periods = periods
        self.offset = offset

    def __call__(self, t):
        """
        Evaluates sine wave at time ``t``.

        :param t: Integer timestep of goal trajectory.
        :return: Sine wave evaluated at time ``t``.
        """
        goal = np.sin(
            self.offset + 2 * np.pi * self.periods * (t / self.steps)
        )
        return shift_and_scale(goal)


class ChirpTask:
    """
    A callable class that implements a chirp function to use as a goal height trajectory. Computes values in ``[-1, 1]``
    and then shifts and scales them into ``[constants.goal_min, constants.goal_max]``.
    """

    def __init__(self, steps=500, periods=1., offset=0.):
        """
        Constructor for ``SineTask`` object.

        :param steps: Length of chirp function.
        :param periods: Number of periods of chirp function.
        :param offset: Chirp function evaluation initial condition. In ``[0, 2 * np.pi]``.
        """
        self.steps = steps
        self.periods = periods
        self.offset = offset

    def __call__(self, t):
        """
        Evaluates chirp function at time ``t``.

        :param t: Integer timestep of goal trajectory.
        :return: Chirp function evaluated at time ``t``.
        """
        goal = np.sin(
            self.offset + 2 * np.pi * self.periods * (t / self.steps) ** 2
        )
        return shift_and_scale(goal)


class StepTask:
    """
    A callable class that implements a step function to use as a trajectory. The step function we use is a sine wave
    rounded to the nearest multiple of ``0.5``, giving values in ``{-1, -0.5, 0, 0.5, 1}``, which are then shifted and
    scaled to the range ``[constants.goal_min, constants.goal_max]``.
    """

    def __init__(self, steps=500, periods=1., offset=0.):
        """
        Constructor for ``SineTask`` object.

        :param steps: Length of step function.
        :param periods: Number of periods of step function.
        :param offset: Step function evaluation initial condition. In ``[0, 2 * np.pi]``.
        """
        self.steps = steps
        self.periods = periods
        self.offset = offset

    def __call__(self, t):
        """
        Evaluates step function at time ``t``.

        :param t: Integer timestep of goal trajectory.
        :return: Step function evaluated at time ``t``.
        """
        goal = np.round(
            np.sin(
                self.offset + 2 * np.pi * self.periods * (t / self.steps)
            ) * 2.
        ) / 2.
        return shift_and_scale(goal)


if __name__ == '__main__':
    steps = 500

    sine_task = SineTask(steps=steps)
    chirp_task = ChirpTask(steps=steps)
    step_task = StepTask(steps=steps)

    import matplotlib.pyplot as plt

    sine_evals, chirp_evals, step_evals = [], [], []
    for t in range(steps):
        sine_evals.append(sine_task(t))
        chirp_evals.append(chirp_task(t))
        step_evals.append(step_task(t))

    fig, axes = plt.subplots(3, 1, sharex='all')
    axes[0].plot(sine_evals)
    axes[1].plot(chirp_evals)
    axes[2].plot(step_evals)

    plt.show()
