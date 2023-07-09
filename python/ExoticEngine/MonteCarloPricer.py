import copy

import numpy as np
from scipy.stats import norm

from ExoticEngine.MarketDataObject import Parameter as P
from ExoticEngine.MonteCarloEngine import ExoticEngine as EE
from ExoticEngine.MonteCarloEngine import SimulationModel as Sim
from ExoticEngine.Payoff import Options as O
from ExoticEngine.Payoff import PathDependentOptions as PDO
from ExoticEngine.Statistics import Statistics as Stats


def BS_CALL(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def BS_PUT(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def vanilla_mc_pricer(
    option: O.VanillaOption,
    rate: P.Parameter,
    sim_model: Sim.Model,
    result_collector: Stats.GetStatistics,
) -> Stats.GetStatistics:
    expiry = option.get_expiry()
    # create a deepy copy to avoid side effects...python passes arg by assignment (reference)
    collector = copy.deepcopy(result_collector)  # not sure if memory efficient
    discount_factor = np.exp(-rate.get_integral(0.0, expiry))
    terminate = False
    spot_t0 = sim_model.get_day0()
    while not terminate:
        discounted_payoff = discount_factor * option.get_payoff(
            spot=sim_model.sde(0, expiry) + spot_t0
        )
        collector.add_one_result(discounted_payoff)
        terminate = collector.terminate()
    return collector


def path_dependent_mc_pricer(
    sim_engine: EE.ExoticEngine, path_count: int = 1000
) -> Stats.GetStatistics:
    # support different termination condition later!
    termination = Stats.ConditionType("PATH_COUNT")
    condition = Stats.TerminationCondition(termination, path_count)
    result_collector = Stats.GetStatistics(condition)
    sim_engine.run_simulation(result_collector, path_count)
    return result_collector
