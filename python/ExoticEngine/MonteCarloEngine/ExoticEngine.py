import abc
import copy
from typing import final

import numpy as np

from ExoticEngine.MarketDataObject import Parameter as P
from ExoticEngine.Payoff import PathDependentOptions as PDO
from ExoticEngine.Statistics import RandomNumberGenerators as RNG
from ExoticEngine.Statistics import Statistics as Stats


class ExoticEngine(abc.ABC):
    def __init__(self, product: PDO.PathDependent, rate: P.Parameter):
        self._product = product
        self._r = rate
        cashflow_times = product.get_possible_cashflow_times()
        self._discounts = []
        for t in cashflow_times:
            self._discounts.append(np.exp(-self._r.get_mean(0, t) * t))

    def run_simulation(self, collector: Stats.GetStatistics, num_paths: int) -> None:
        for i in range(num_paths):
            simulated_spots = self.run_pathwise_simulation()
            discounted_flows = self.get_pathwise_discounted_flow(simulated_spots)
            collector.add_one_result(discounted_flows)

    @abc.abstractmethod
    def run_pathwise_simulation(self) -> tuple[float]:
        """This is an abstract method: dynamics is model dependent"""
        pass

    def get_pathwise_discounted_flow(self, simulated_spots: tuple[float]) -> float:
        future_flows = self._product.get_cash_flows(simulated_spots)
        total_discounted_flows = 0
        # I am sure there is a more Pythonic way of doing this running sum
        for i, cf in enumerate(future_flows):
            total_discounted_flows += cf.get_amount() * self._discounts[i]
        return total_discounted_flows


@final
class ExoticBSEngine(ExoticEngine):
    def __init__(
        self,
        product: PDO.PathDependent,
        rate: P.Parameter,
        dividend: P.Parameter,
        repo: P.Parameter,
        vol: P.Parameter,
        generator: RNG.RandomBase,
        spot: float,
    ):
        super().__init__(product, rate)
        self._ref_times = product.get_reference_times()
        self._num_ref_times = len(self._ref_times)
        self._RNG = copy.deepcopy(generator)  # might not need deep copy
        self._RNG.reset_dimension(self._num_ref_times)
        self._spot = spot

        variances, drifts = [], []
        for i in range(self._num_ref_times - 1):
            t = self._ref_times[i]
            t_next = self._ref_times[i + 1]
            v = vol.get_square_integral(t, t_next)
            variances.append(v)
            drifts.append(
                rate.get_integral(t, t_next)
                - dividend.get_integral(t, t_next)
                - repo.get_integral(t, t_next)
                - 0.5 * v
            )
        self._variances = tuple(variances)
        self._drifts = tuple(drifts)

    def run_pathwise_simulation(self):
        std_normals = self._RNG.get_gaussian()  # not sure if this should be in ctor
        current_log_spot = np.log(self._spot)
        spot_values = []
        for i in range(self._num_ref_times - 1):
            current_log_spot += (
                self._drifts[i] + np.sqrt(self._variances[i]) * std_normals[i]
            )
            spot_values.append(np.exp(current_log_spot))
        return tuple(spot_values)
