import numpy as np
import matplotlib.pyplot as plt
from observe import observe


class MackeyGlassGenerator:
    def __init__(self, mu=1, beta=2, tau=2, n=9.65, series_padding=2):
        self._mu = mu
        self._beta = beta
        self._tau = tau
        self._n = n
        self._series_padding = series_padding

    def _step(self, x, x_tau):
        dx = self._beta * (x_tau / (1 + x_tau ** self._n)) - self._mu * x
        return x + dx

    def generate_series(self, length, previous_x_record=None):
        """
        Generate a Mackey Glass series of length steps, if previous_x_record is not set then the start of the
        series will be randomly initialized. Otherwise the series generation begins at the end of the supplied
        previous_x_record.

        Args:
            length (int): The number of steps to generate data for.
            previous_x_record (ndarray, optional):

        Returns:
            ndarray

        """
        x_record = np.zeros((length, 1))

        if previous_x_record is None:
            # randomly initializing x_record values
            previous_x_record = np.random.uniform(low=0, high=1.4, size=(self._tau * self._series_padding, 1))
        elif previous_x_record.shape[0] < self._tau:
            raise ValueError('previous_x_record must have a length greater than or equal to tau')

        primed_x_record = np.concatenate([previous_x_record, x_record])

        for step in range(0, np.shape(primed_x_record)[0]):
            if step >= previous_x_record.shape[0]:
                primed_x_record[step] = self._step(x=primed_x_record[step - 1],
                                                   x_tau=primed_x_record[step - 1 - self._tau])

        # slicing off the previous series of x values
        x_record = primed_x_record[self._tau * self._series_padding:-1:1]

        return x_record


def main():
    generator = MackeyGlassGenerator()

    first_series_length = 1000
    start_x_record = generator.generate_series(first_series_length)
    observed_start_x_record = observe(start_x_record, std=0.00000001)

    final_series_length = 10000
    x_record = generator.generate_series(length=final_series_length,
                                         previous_x_record=start_x_record)

    x_record_from_observations = generator.generate_series(length=final_series_length,
                                                           previous_x_record=observed_start_x_record)

    plt.plot(x_record, label='x')
    plt.plot(x_record_from_observations, label='x from observations')
    plt.plot(np.abs(x_record - x_record_from_observations), label='difference')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
