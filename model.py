import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
path = "Data/tvmarketing.csv"
adv = pd.read_csv(path)

# Extract features and target
X = adv['TV'].values.reshape(-1, 1)
Y = adv['Sales'].values

# Train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X, Y)

# Save the trained model using joblib
joblib.dump(lr_model, 'lr_model.joblib')