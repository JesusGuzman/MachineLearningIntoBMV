import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from financial_features import generate_random_features
from indicators import *
from labels import labels
from sklearn.feature_selection import RFE

# Load OHLCV data into a DataFrame
df = pd.read_csv('ohlcv_data.csv', index_col='Date', parse_dates=True)
y_label = labels(df.copy())

rsi_features = calculate_rsi(df.copy())
macd_features = calculate_macd(df.copy())
stochastic_features = calculate_stochastic(df.copy())
vix_features = calculate_vix(df.copy())
b_bands_features = calculate_bollinger_bands(df.copy())
atr_features = calculate_atr(df.copy())
rsi_s_features = calculate_rsi_smoothed(df.copy())
ma_features = calculate_ma(df.copy())
ema_features = calculate_ema(df.copy())
adx_features = calculate_adx(df.copy())
obv_features = calculate_obv(df.copy())
roc_features = calculate_roc(df.copy())
williams_features = calculate_williams_r(df.copy())

random_features = generate_random_features(df.copy())

dataset = pd.concat([random_features,rsi_features,macd_features,
                     stochastic_features,vix_features,b_bands_features,
                     atr_features,rsi_s_features,ma_features,
                     ema_features,adx_features,obv_features,
                     roc_features,williams_features,
                     y_label], axis=1)

dataset = dataset.dropna(axis=0)

print(dataset.shape)

# Separate features and labels
X = dataset.drop(['label'], axis=1)
y = dataset[['label']]

# Initialize the random forest classifier
rf = RandomForestClassifier()

np.int = np.int32
np.float = np.float64
np.bool = np.bool_

# Fit the classifier to the data
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_

# Get sorted indices of feature importances
sorted_indices = np.argsort(importances)[::-1]

# Print feature importance ranking
print("Feature Importance Ranking:")
for i in sorted_indices:
    print(X.columns[i], importances[i])


print('Feature Importance Boruta:')
# Initialize the Boruta feature selection
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
boruta = BorutaPy(rf, n_estimators='auto', verbose=2)

# Separate features and labels
X_ = dataset.drop(['label'], axis=1)
y_ = dataset[['label']]

X = X_.values
y = y_.values
y = y.ravel()

# Perform feature selection
boruta.fit(X, y)

# Get selected feature indices
selected_indices = boruta.support_

# Get selected feature names
selected_features = X_.columns[selected_indices]

# Print selected feature names
print("Selected Features:")
for feature in selected_features:
    print(feature)


print('Feature Importance Recursive Feature Elimination:')

# Initialize the random forest classifier
rf = RandomForestClassifier()

# Initialize the RFE feature selection
rfe = RFE(rf, n_features_to_select=5)

# Perform feature selection
rfe.fit(X, y)

# Get selected feature indices
selected_indices_rfe = rfe.support_

# Get selected feature names
selected_features_rfe = X_.columns[selected_indices_rfe]

# Print selected feature names
print("Selected Features (RFE):")
for feature in selected_features_rfe:
    print(feature)