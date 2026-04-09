import pandas as pd
import numpy as np

def crossed_above(series1, series2):
    """
    Returns a boolean series where series1 crossed above series2.
    """
    if isinstance(series1, (pd.Series, pd.DataFrame)) and isinstance(series2, (pd.Series, pd.DataFrame)):
        return (series1 > series2) & (series1.shift(1) <= series2.shift(1))
    return series1 > series2 # Fallback for scalar/float evaluation

def crossed_below(series1, series2):
    """
    Returns a boolean series where series1 crossed below series2.
    """
    if isinstance(series1, (pd.Series, pd.DataFrame)) and isinstance(series2, (pd.Series, pd.DataFrame)):
        return (series1 < series2) & (series1.shift(1) >= series2.shift(1))
    return series1 < series2 # Fallback for scalar/float evaluation

# Aliases for variations in strategy code
cross_above = crossed_above
cross_below = crossed_below
crossed_up = crossed_above
crossed_down = crossed_below
