
import pandas as pd
import matplotlib.pyplot as plt

# Backtest using DataFrame prices and signals
def backtest(prices, signals):
    # Initialize the portfolio and positions
    portfolio = pd.DataFrame(index=prices.index)
    portfolio['Position'] = signals['Buy'].astype(int) - signals['Sell'].astype(int)
    # Calculate the daily returns
    portfolio['Returns'] = prices['Close'].pct_change()
    # Calculate the strategy returns
    portfolio['StrategyReturns'] = portfolio['Returns'] * portfolio['Position'].shift()
    # Calculate the cumulative returns
    portfolio['CumulativeReturns'] = (1 + portfolio['StrategyReturns']).cumprod()
    return portfolio
    
def plot_backtest(portfolio):
    # Plot the cumulative returns
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio['CumulativeReturns'])
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.grid(True)
    plt.show()

# Calculate metrics of backtest
def calculate_metrics(portfolio):
    # Calculate daily returns
    
    portfolio['Return'] = portfolio['StrategyReturns']

    # Calculate cumulative returns
    portfolio['Cumulative Return'] = portfolio['CumulativeReturns'] 

    # Calculate maximum drawdown
    portfolio['Rolling Max'] = portfolio['Cumulative Return'].rolling(window=len(portfolio), min_periods=1).max()
    portfolio['Drawdown'] = portfolio['Cumulative Return'] / portfolio['Rolling Max'] - 1
    max_drawdown = portfolio['Drawdown'].min()

    # Calculate annualized return
    start_date = portfolio.index[0]
    end_date = portfolio.index[-1]
    num_years = (end_date - start_date).days / 365
    annualized_return = (portfolio['Cumulative Return'].iloc[-1]) ** (1 / num_years) - 1

    # Calculate annualized volatility
    annualized_volatility = portfolio['Return'].std() * (252 ** 0.5)

    # Calculate Sharpe ratio
    risk_free_rate = 0.02  # Assuming a risk-free rate of 2%
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    # Calculate total trades
    total_trades = len(portfolio[portfolio['Position'] != 0])

    # Calculate win rate
    win_trades = len(portfolio[(portfolio['Position'] == 1) & (portfolio['Return'] > 0)])
    win_rate = win_trades / total_trades

    # Calculate average return per trade
    average_return_per_trade = portfolio['Return'].mean()

    # Return the calculated metrics
    return max_drawdown, annualized_return, annualized_volatility, sharpe_ratio, total_trades, win_rate, average_return_per_trade