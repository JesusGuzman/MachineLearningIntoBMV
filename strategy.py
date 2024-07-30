import pandas as pd
import warnings
#from financial_features import *
from indicators import calculate_rsi, calculate_stochastic, calculate_williams_r

warnings.filterwarnings("ignore")

# Implement your strategy here
def strategy_rsi(data):
    rsi = calculate_rsi(data)['rsi']

    buy_signal = (rsi < 30)
    sell_signal = (rsi > 70)

    # Return the buy and sell signals
    signals = pd.DataFrame({'Buy': buy_signal, 'Sell': sell_signal})
    return signals

def strategy_stochastic(data):
    # Calculate the stochastic oscillator
    df = calculate_stochastic(data)

    # Generate buy and sell signals based on the stochastic oscillator
    buy_signal = (df['stochastic_K'] < 20) & (df['stochastic_D'] < 20)
    sell_signal = (df['stochastic_K'] > 80) & (df['stochastic_D'] > 80)

    # Return the buy and sell signals
    signals = pd.DataFrame({'Buy': buy_signal, 'Sell': sell_signal})
    return signals

def strategy_williams_r(data):
    # Calculate the Williams %R
    williams = calculate_williams_r(data)['williams_r']

    # Generate buy and sell signals based on the Williams %R
    buy_signal = (williams < -80)
    sell_signal = (williams > -20)

    # Return the buy and sell signals
    signals = pd.DataFrame({'Buy': buy_signal, 'Sell': sell_signal})
    return signals