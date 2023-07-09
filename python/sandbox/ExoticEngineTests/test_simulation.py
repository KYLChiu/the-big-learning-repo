import copy

import numpy as np

from ExoticEngine.MonteCarloEngine import ExoticEngine as EE
from ExoticEngine.MonteCarloEngine import SimulationModel as Sim
from ExoticEngine.Payoff import Options as O
from ExoticEngine.Payoff import PathDependentOptions as PDO
from ExoticEngine.Statistics import RandomNumberGenerators as RNG
from sandbox.ExoticEngineTests import pricer_helper as helper


def test_vanilla_BS_simulation_zero_rate_zero_vol():
    param_dict = helper.build_constant_market_param(
        rate=0.0, vol=0.0, repo=0.0, div=0.0
    )
    RNGenerator = RNG.TestRandom(dimension=1)
    spot = 100
    EqModel = Sim.BSModel(
        spot=spot,
        r=param_dict["Rate"],
        repo_rate=param_dict["Repo"],
        div_yield=param_dict["Div"],
        vol=param_dict["Vol"],
        RNG=RNGenerator,
    )
    assert EqModel.sde(0, 5) == 0


def test_vanilla_BS_simulation_zero_vol(tolerance=1e-8):
    param_dict = helper.build_constant_market_param(
        rate=0.05, vol=0.0, repo=0.03, div=0.0
    )
    RNGenerator = RNG.TestRandom(dimension=1)
    maturity = 5.0
    spot = 100
    EqModel = Sim.BSModel(
        spot=spot,
        r=param_dict["Rate"],
        repo_rate=param_dict["Repo"],
        div_yield=param_dict["Div"],
        vol=param_dict["Vol"],
        RNG=RNGenerator,
    )
    discount_factor_r = np.exp(-param_dict["Rate"].get_mean(0, maturity) * maturity)
    discount_factor_q = np.exp(-param_dict["Repo"].get_mean(0, maturity) * maturity)
    discount_factor_d = np.exp(-param_dict["Div"].get_mean(0, maturity) * maturity)
    effective_DF = discount_factor_r / discount_factor_q / discount_factor_d
    assert abs(EqModel.sde(0, maturity) - spot * (1 / effective_DF - 1)) < tolerance


def test_asian_BS_simulation_zero_vol(tolerance=1e-8):
    num_stopping_dates = 9
    ref_times = tuple(i for i in range(num_stopping_dates))
    delivery_time = 9
    call = O.PayOffCall(90)
    RNGenerator = RNG.TestRandom(dimension=1)
    asian = PDO.AsianOption(
        reference_times=ref_times, delivery_time=delivery_time, payoff=call
    )
    assert asian.get_reference_times() == ref_times
    param_dict = helper.build_constant_market_param(
        rate=0.05, vol=0.0, repo=0.01, div=0.02
    )
    spot = 100
    BSModel = EE.ExoticBSEngine(
        product=asian,
        rate=param_dict["Rate"],
        dividend=param_dict["Div"],
        repo=param_dict["Repo"],
        vol=param_dict["Vol"],
        generator=RNGenerator,
        spot=spot,
    )
    simulated_spots = BSModel.run_pathwise_simulation()
    assert len(simulated_spots) == len(ref_times) - 1

    expected_spots = []
    new_spot = copy.deepcopy(spot)
    for i in range(num_stopping_dates - 1):
        t = ref_times[i]
        t_next = ref_times[i + 1]
        effective_growth = (
            param_dict["Rate"].get_integral(t, t_next)
            - param_dict["Div"].get_integral(t, t_next)
            - param_dict["Repo"].get_integral(t, t_next)
        )
        new_spot *= np.exp(effective_growth)
        expected_spots.append(new_spot)
        assert abs(simulated_spots[i] - expected_spots[i]) < tolerance
