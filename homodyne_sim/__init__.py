import utils 
import Apparatus 
import Simulation 

__version__ = (0, 1, 0)

__modules = [utils, Apparatus, Simulation]
map(reload, __modules)

__all__ = reduce(lambda a, b: a+b, map(lambda mod: mod.__all__, __modules)) + ['__version__']
