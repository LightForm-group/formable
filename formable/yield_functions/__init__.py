"""`formable.yield_functions.__init__.py`"""

from formable.yield_functions.base import (
    YieldFunction, fitting_callable, DEF_2D_RES, DEF_3D_RES)
from formable.yield_functions.yield_functions import *

YIELD_FUNCTION_MAP = {
    'VonMises':             VonMises,
    'Tresca':               Tresca,
    'Hosford':              Hosford,
    'Barlat_Yld91':         Barlat_Yld91,
    'Barlat_Yld2000_2D':    Barlat_Yld2000_2D,
    'Barlat_Yld2004_18p':   Barlat_Yld2004_18p,
    'Hill1979':             Hill1979,
    'Hill1948':             Hill1948,
    'Dummy':                Dummy,
}

AVAILABLE_YIELD_FUNCTIONS = list(YIELD_FUNCTION_MAP.keys())
