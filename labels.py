import pandas as pd

def labels(df):
    # Calculate momentum indicator
    df['label'] = df['Close'].pct_change()

    # Define buy and sell signals
    df['up_signal'] = df['label'] > 0
    df['down_signal'] = df['label'] < 0

    # Initialize position
    position = 0

    # Iterate over the DataFrame
    for i in range(1, len(df)):
        # Check for buy signal
        if df['up_signal'].iloc[i] and position == 0:
            position = 1
        # Check for sell signal
        elif df['down_signal'].iloc[i] and position == 1:
            position = 0

        # Assign position value to the DataFrame
        df['label'].iloc[i] = position
    
    df['returns'] = df['Close'].pct_change()
    # Replace 0.0 with -1.0 in the 'label' column
    df['label'] = df['label'].replace(0.0, -1.0)
    
    return df[['label']]