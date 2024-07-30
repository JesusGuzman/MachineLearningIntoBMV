import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('features.csv', index_col='Date', parse_dates=True)
data = data.dropna(axis=0)
print(data)

# Filter the data for specific columns
columns_to_keep = [
'%D',
'CumulativeReturns',
'Volume',
'Adj Close',
'High',
'Close',
'Low',
'%K',
'Open',
'Returns',
'StrategyReturns',
'stochastic'
]
data_filtered = data[columns_to_keep]
print(data_filtered)

# Split the data into training, validation, and evaluation sets
train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

import pdb; pdb.set_trace()

# Prepare the features and target variables
X_train = train_data.drop('LABEL', axis=1)
y_train = train_data['LABEL']
X_val = val_data.drop('LABEL', axis=1)
y_val = val_data['LABEL']
X_eval = eval_data.drop('LABEL', axis=1)
y_eval = eval_data['LABEL']

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Validate the model
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy}")

# Evaluate the model
y_eval_pred = model.predict(X_eval)
eval_accuracy = accuracy_score(y_eval, y_eval_pred)
print(f"Evaluation Accuracy: {eval_accuracy}")

print(y_eval_pred)