import abc
from typing import final

import numpy as np

from ExoticEngine.MarketDataObject import Parameter as P
from ExoticEngine.Statistics import RandomNumberGenerators as RNG


class Model(abc.ABC):
    @abc.abstractmethod
    def sde(self, t1: float, t2: float) -> float:
        pass

    @abc.abstractmethod
    def get_day0(self):
        pass


@final
class BSModel(Model):
    def __init__(
        self,
        spot: float,
        r: P.Parameter,
        repo_rate: P.Parameter,
        div_yield: P.Parameter,
        vol: P.Parameter,
        RNG: RNG.RandomBase,
    ):
        """
        Assume deterministic parameters, r,q,d,sigma
        Smile captured if local vol
        Probably should take in parameters as "Model" instances - to support potential stochastic rate/vol
        """
        self._spot = spot
        self._r = r
        self._q = repo_rate
        self._d = div_yield
        self._vol = vol
        self._RNG = RNG

    def get_day0(self):
        return self._spot

    def sde(self, t1: float, t2: float) -> float:
        """
        returns: dS = S(t2) - S(t1)
        S(t2) = S(t1) * exp[ int {r(s) -q(s) -d(s) -0.5 sigma^2(s)} ds + int sigma(s) dW ]
        single path only...would it be slow?
        Assumes risk-neutral measure
        """
        assert t2 > t1
        assert t1 >= 0
        dt = t2 - t1
        ito = 0.5 * self._vol.get_root_mean_square(t1, t2) ** 2
        drift = (
            self._r.get_mean(t1, t2)
            - self._q.get_mean(t1, t2)
            - self._d.get_mean(t1, t2)
            - ito
        ) * dt
        # recall by Ito isometry: Ito integral of f(t) has variance int f(t)^2 dt
        diffusion = (
            self._vol.get_square_integral(t1, t2)
            * self._RNG.get_gaussian()[0]  # bad design?
        )
        return self._spot * (np.exp(drift + diffusion) - 1)
