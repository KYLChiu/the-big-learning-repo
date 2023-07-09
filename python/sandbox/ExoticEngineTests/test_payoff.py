import numpy as np

from ExoticEngine.Payoff import Options as O
from ExoticEngine.Payoff import PathDependentOptions as PDO


def test_call_payoff():
    call = O.PayOffCall(strike=1.0)
    assert call.payoff(spot=1.0) == 0
    assert call.payoff(spot=0.5) == 0
    assert call.payoff(spot=1.5) == 0.5


def test_put_payoff():
    put = O.PayOffPut(strike=3.0)
    assert put.payoff(spot=1.0) == 2.0
    assert put.payoff(spot=0.5) == 2.5
    assert put.payoff(spot=5.5) == 0.0


def test_double_digital_payoff():
    double_digital = O.PayOffDoubleDigital(upper=30.0, lower=20.0)
    assert double_digital.payoff(spot=25) == 1
    assert double_digital.payoff(spot=0.5) == 0
    assert double_digital.payoff(spot=50) == 0.0


def test_put_option_payoff():
    put = O.PayOffPut(strike=3.0)
    put_option = O.VanillaOption(put, 5.0)
    assert put_option.get_payoff(spot=1.0) == 2.0
    assert put_option.get_payoff(spot=6.0) == 0.0


def test_option_expiry():
    call = O.PayOffCall(strike=3.0)
    expiry = 5.0
    call_option = O.VanillaOption(call, expiry)
    assert call_option.get_expiry() == expiry


def test_asian_call_payoff():
    spots = tuple(i**2 for i in np.arange(0, 21))
    delivery_time = 0.5 * len(spots)
    ref_times = tuple(np.arange(0, delivery_time, 0.5))
    Ks = [60, 100, 130, 150, 180]
    for K in Ks:
        call = O.PayOffCall(strike=K)
        asian = PDO.AsianOption(ref_times, delivery_time, call)
        assert asian.get_max_number_of_cashflows() == 1
        assert asian.get_possible_cashflow_times()[0] == delivery_time
        assert asian.get_reference_times() == ref_times
        assert asian.get_cash_flows(spots)[0].get_amount() == max(np.mean(spots) - K, 0)
        assert asian.get_cash_flows(spots)[0].get_time() == delivery_time


def test_asian_put_payoff():
    spots = tuple(i**2 for i in np.arange(0, 21))
    delivery_time = 0.5 * len(spots)
    ref_times = tuple(np.arange(0, delivery_time, 0.5))
    Ks = [60, 100, 130, 150, 180]
    for K in Ks:
        put = O.PayOffPut(strike=K)
        asian = PDO.AsianOption(ref_times, delivery_time, put)
        assert asian.get_max_number_of_cashflows() == 1
        assert asian.get_possible_cashflow_times()[0] == delivery_time
        assert asian.get_reference_times() == ref_times
        assert asian.get_cash_flows(spots)[0].get_amount() == max(K - np.mean(spots), 0)
        assert asian.get_cash_flows(spots)[0].get_time() == delivery_time
