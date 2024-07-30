import pandas as pd
import numpy as np
from indicators import calculate_rsi

def generate_random_features(df):
    # Generate 10 columns with random 0s and 1s
    for i in range(10):
        column_name = f'random_{i+1}'
        df[column_name] = np.random.randint(2, size=len(df))
        
    return df.drop(['Open','High','Low','Close','Adj Close','Volume'], axis=1)
