from __future__ import absolute_import
from six.moves import map
from functools import reduce
from importlib import reload

from . import utils as _u
from . import Apparatus as _a
from . import Simulation as _s

__version__ = (0, 1, 0)

__modules = [_u, _a, _s]
map(reload, __modules)

from .utils import *
from .Apparatus import *
from .Simulation import *

__all__ = reduce(lambda a, b: a+b, map(lambda mod: mod.__all__, __modules)) + ['__version__']
