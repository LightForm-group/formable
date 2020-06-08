def get_yield_function_map():
    from formable.yielding.yield_functions import (
        VonMises,
        Tresca,
        Hosford,
        Barlat_Yld91,
        Barlat_Yld2000_2D,
        Barlat_Yld2004_18p,
        Hill1979,
        Hill1948,
        Dummy2DYieldFunction,
    )
    YIELD_FUNCTION_MAP = {
        'VonMises':             VonMises,
        'Tresca':               Tresca,
        'Hosford':              Hosford,
        'Barlat_Yld91':         Barlat_Yld91,
        'Barlat_Yld2000_2D':    Barlat_Yld2000_2D,
        'Barlat_Yld2004_18p':   Barlat_Yld2004_18p,
        'Hill1979':             Hill1979,
        'Hill1948':             Hill1948,
        'Dummy2DYieldFunction': Dummy2DYieldFunction,
    }
    return YIELD_FUNCTION_MAP


def get_available_yield_functions():
    return list(get_yield_function_map().keys())
