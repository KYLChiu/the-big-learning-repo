import abc
from enum import Enum
from typing import final

import numpy as np


# this might be useless
class RandomNumberType(Enum):
    NUMPY = "NUMPY"
    PSEUDO_RANDOM = "PSEUDO_RANDOM"
    SOBOL = "SOBOL"


class RandomBase(abc.ABC):
    def __init__(self, dimension: int = 1):
        self._dim = dimension

    @abc.abstractmethod
    def get_uniforms(self):
        pass

    # @abc.abstractmethod
    def set_seed(self):
        # CY: do I need to return this and ideally where should I define the seed?
        return np.random.seed(1)

    def get_gaussian(self):
        """
        returns a vector of standard normal rv's
        """
        return np.random.normal(loc=0, scale=1, size=self._dim)

    def reset_dimension(self, dimension: int):
        if dimension <= 0:
            raise ValueError(f"Check dimension input: dimension={dimension}")
        self._dim = dimension


@final
class TestRandom(RandomBase):
    def __init__(self, dimension: int = 1):
        """Dummy class: need to be renamed!"""
        super().__init__(dimension=dimension)

    def get_uniforms(self):
        raise Exception("Not implemented")

