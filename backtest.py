import yfinance as yf
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from utils import backtest, plot_backtest, calculate_metrics
from strategy import strategy_rsi, strategy_stochastic, strategy_williams_r

warnings.filterwarnings("ignore")

# Define the ticker symbol and time range
ticker = "CUERVO.MX"
#start_date = "2019-01-01"
#end_date = "2022-04-31"

start_date = "2022-11-30" #VAL
end_date = "2023-6-30" #VAL

# Download the OHLCV data
data = yf.download(ticker, start=start_date, end=end_date)
# Save the data as a CSV file
data.to_csv("ohlcv_data.csv")

# Call the strategy implementation function
#signals = strategy_rsi(data)
#signals = strategy_stochastic(data)
print('usando williams')
signals = strategy_williams_r(data)

# Save the buy and sell signals as a CSV file
signals.to_csv("output/signals.csv", index=True)

# Call the backtest function
portfolio = backtest(data, signals)

# Save the portfolio as a CSV file with index
portfolio.to_csv("output/portfolio.csv", index=True)

# Call the calculate_metrics function
max_drawdown, annualized_return, annualized_volatility, sharpe_ratio, total_trades, win_rate, average_return_per_trade = calculate_metrics(portfolio)

# Print the calculated metrics
print("Max Drawdown:", max_drawdown)
print("Annualized Return:", annualized_return)
print("Annualized Volatility:", annualized_volatility)
print("Sharpe Ratio:", sharpe_ratio)
print("Total Trades:", total_trades)
print("Win Rate:", win_rate)
print("Average Return per Trade:", average_return_per_trade)

plot_backtest(portfolio)
