import pandas as pd

def calculate_rsi(data, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given dataset.

    Parameters:
    - data: pandas DataFrame or Series containing the 'Close' prices.
    - period: int, optional (default=14)
        The number of periods to consider for calculating the RSI.

    Returns:
    - rsi: pandas Series
        The calculated RSI values.

    """
    delta = data['Close'].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = pd.DataFrame({'rsi': rsi})
    return rsi

# Calculate Stochastic Oscillator
def calculate_stochastic(data, k_period=14, d_period=3):
    """
    Calculate the stochastic oscillator for a given dataset.

    Parameters:
    - data (pandas.DataFrame): The dataset containing 'High', 'Low', and 'Close' columns.
    - k_period (int): The period length for calculating the %K line (default: 14).
    - d_period (int): The period length for calculating the %D line (default: 3).

    Returns:
    - df (pandas.DataFrame): A DataFrame containing the calculated %K and %D values.

    """
    lowest_low = data['Low'].rolling(window=k_period).min()
    highest_high = data['High'].rolling(window=k_period).max()
    k = ((data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
    d = k.rolling(window=d_period).mean()
    df = pd.DataFrame({'stochastic_K': k, 'stochastic_D': d})
    return df

# Calculate MACD
def calculate_macd(data, fast_period=12, slow_period=25, signal_period=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) indicator.

    Parameters:
    - data: DataFrame containing the 'Close' column.
    - fast_period: Integer representing the fast EMA period (default: 12).
    - slow_period: Integer representing the slow EMA period (default: 25).
    - signal_period: Integer representing the signal EMA period (default: 9).

    Returns:
    - DataFrame with columns 'macd', 'macd_signal', and 'macd_histogram'.
    """
    exp1 = data['Close'].ewm(span=fast_period, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    df = pd.DataFrame({'macd': macd, 'macd_signal': signal, 'macd_histogram': histogram})
    return df

# Calculate VIX
def calculate_vix(data):
    """
    Calculate the VIX (Volatility Index) based on the given data.

    Parameters:
    - data: pandas DataFrame containing the necessary columns (High, Low, Close) for calculation.

    Returns:
    - vix: pandas DataFrame containing the calculated VIX values.

    """
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)
    atr = tr.rolling(window=14).mean()
    vix = atr / data['Close'] * 100
    vix = pd.DataFrame({'vix': vix})
    return vix

# Calculate Bollinger Bands
def calculate_bollinger_bands(data, period=20, std_dev=2):
    """
    Calculate the Bollinger Bands for a given dataset.

    Parameters:
    - data (DataFrame): The input dataset containing the 'Close' column.
    - period (int): The number of periods to consider for the rolling mean and standard deviation. Default is 20.
    - std_dev (int): The number of standard deviations to use for the upper and lower bands. Default is 2.

    Returns:
    - bollinger_bands (DataFrame): A DataFrame containing the upper and lower Bollinger Bands.
    """
    rolling_mean = data['Close'].rolling(window=period).mean()
    rolling_std = data['Close'].rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    bollinger_bands = pd.DataFrame({'upper_band': upper_band, 'lower_band': lower_band})
    return bollinger_bands

# Calculate Average True Range (ATR)
def calculate_atr(data, period=14):
    """
    Calculate the Average True Range (ATR) for a given dataset.

    Parameters:
    - data (pandas.DataFrame): The input dataset containing 'High', 'Low', and 'Close' columns.
    - period (int): The number of periods to consider for calculating the ATR. Default is 14.

    Returns:
    - atr (pandas.DataFrame): A DataFrame containing the calculated ATR values.

    """
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)
    atr = tr.rolling(window=period).mean()
    atr = pd.DataFrame({'atr': atr})
    return atr

# Calculate Relative Strength Index (RSI) Smoothed
def calculate_rsi_smoothed(data, period=14, smoothing_period=3):
    """
    Calculate the smoothed Relative Strength Index (RSI) for a given dataset.

    Parameters:
    - data (pandas.DataFrame): The dataset containing the 'Close' prices.
    - period (int): The number of periods to consider when calculating the RSI. Default is 14.
    - smoothing_period (int): The number of periods to use for smoothing the RSI. Default is 3.

    Returns:
    - rsi_smoothed (pandas.DataFrame): The smoothed RSI values.

    """
    delta = data['Close'].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_smoothed = rsi.rolling(window=smoothing_period).mean()
    rsi_smoothed = pd.DataFrame({'rsi_smoothed': rsi_smoothed})
    return rsi_smoothed


# Calculate Moving Average (MA)
def calculate_ma(data, period=20):
    ma = data['Close'].rolling(window=period).mean()
    ma = pd.DataFrame({'ma': ma})
    return ma

# Calculate Exponential Moving Average (EMA)
def calculate_ema(data, period=20):
    ema = data['Close'].ewm(span=period, adjust=False).mean()
    ema = pd.DataFrame({'ema': ema})
    return ema

# Calculate Average Directional Index (ADX)
def calculate_adx(data, period=14):
    """
    Calculate the Average Directional Index (ADX) for a given dataset.

    Parameters:
    - data (pandas.DataFrame): The input dataset containing 'High' and 'Low' columns.
    - period (int): The period over which to calculate the ADX. Default is 14.

    Returns:
    - adx (pandas.DataFrame): The calculated ADX values.

    """
    tr = data['High'] - data['Low']
    tr_pos = (data['High'] - data['High'].shift()).clip(lower=0)
    tr_neg = (data['Low'].shift() - data['Low']).clip(lower=0)
    tr_pos_avg = tr_pos.rolling(window=period).mean()
    tr_neg_avg = tr_neg.rolling(window=period).mean()
    tr_pos_avg = tr_pos_avg.fillna(0)
    tr_neg_avg = tr_neg_avg.fillna(0)
    tr_pos_avg = tr_pos_avg.rolling(window=period).mean()
    tr_neg_avg = tr_neg_avg.rolling(window=period).mean()
    dx = (abs(tr_pos_avg - tr_neg_avg) / (tr_pos_avg + tr_neg_avg)) * 100
    adx = dx.rolling(window=period).mean()
    adx = pd.DataFrame({'adx': adx})
    return adx

# Calculate On-Balance Volume (OBV)
def calculate_obv(data):
    obv = pd.Series(0, index=data.index)
    obv[data['Close'] > data['Close'].shift()] = data['Volume']
    obv[data['Close'] < data['Close'].shift()] = -data['Volume']
    obv = obv.cumsum()
    obv = pd.DataFrame({'obv': obv})
    return obv

# Calculate Rate of Change (ROC)
def calculate_roc(data, period=12):
    roc = (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period) * 100
    roc = pd.DataFrame({'roc': roc})
    return roc

# Calculate Williams %R
def calculate_williams_r(data, period=14):
    highest_high = data['High'].rolling(window=period).max()
    lowest_low = data['Low'].rolling(window=period).min()
    wr = (highest_high - data['Close']) / (highest_high - lowest_low) * -100
    wr = pd.DataFrame({'williams_r': wr})
    return wr
