from ExoticEngine.MarketDataObject import Parameter as P
from ExoticEngine.Statistics import RandomNumberGenerators as RNG
from ExoticEngine.Statistics import Statistics as Stats


def build_constant_market_param(rate: float, vol: float, repo: float, div: float):
    Rate = P.Parameter(param=lambda t: rate)
    Vol = P.Parameter(param=lambda t: vol)
    Repo = P.Parameter(param=lambda t: repo)
    Div = P.Parameter(param=lambda t: div)
    return {"Rate": Rate, "Vol": Vol, "Repo": Repo, "Div": Div}


def build_collector(condition_type: str, criteria: float):
    condition = Stats.ConditionType(condition_type)
    termination_condition = Stats.TerminationCondition(condition, criteria)
    return Stats.GetStatistics(termination_condition)
