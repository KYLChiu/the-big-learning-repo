import abc

import numpy as np

from ExoticEngine.Payoff import Options as O


class CashFlow:
    def __init__(self, time: float, amount: float):
        self._time = time
        self._amount = amount

    def set_amount(self, amount):
        self._amount = amount

    def get_amount(self):
        return self._amount

    def get_time(self):
        return self._time


class PathDependent(abc.ABC):
    def __init__(self, reference_times: tuple[float]):
        self._ref_times = reference_times

    def get_reference_times(self) -> tuple[float]:
        return self._ref_times

    @abc.abstractmethod
    def get_max_number_of_cashflows(self) -> int:
        pass

    @abc.abstractmethod
    def get_possible_cashflow_times(self) -> tuple[float]:
        pass

    @abc.abstractmethod
    def get_cash_flows(self, spot_values: tuple[float]) -> tuple[CashFlow]:
        pass


class AsianOption(PathDependent):
    def __init__(
        self, reference_times: tuple[float], delivery_time: float, payoff: O.PayOff
    ):
        self._ref_times = reference_times
        self._delivery_time = delivery_time
        self._payoff = payoff

    def get_max_number_of_cashflows(self) -> int:
        return 1

    def get_possible_cashflow_times(self) -> tuple[float]:
        return (self._delivery_time,)

    def get_cash_flows(self, spot_values: tuple[float]) -> tuple[CashFlow]:
        cash_flow = CashFlow(
            self._delivery_time, self._payoff.payoff(float(np.mean(spot_values)))
        )
        return (cash_flow,)
