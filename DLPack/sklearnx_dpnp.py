# Test interoperability of DPNP with SKLearnX

import numpy as np
import dpnp as dp
import dpctl.tensor as dpt
from sklearnex import patch_sklearn, config_context
patch_sklearn()
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

# Create input and output data on GPU with dpnp
print("")
X = dp.random.rand(10000,6)*10.0 - 5.0 # Create random inputs in [-5,5) range
y = dp.random.rand(10000,1)*2.0 - 1.0 # Create random outputs in [1,1) range
print(f"Created DPNP arrays on device {X.device} \n")

# Split data into train and validation sets
with config_context(target_offload="gpu:0"):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
print(f"train_test_split produced outputs of type {type(X_train)}\n")

# Move data back to GPU
X_train = dpt.from_numpy(X_train)
X_val = dpt.from_numpy(X_val)
y_train = dpt.from_numpy(y_train)
y_val = dpt.from_numpy(y_val)
print(f"Moved split data back to {X_train.device} \n")

# Normalize the data
#with config_context(target_offload="gpu:0"):
#    scaler_X = MinMaxScaler() 
#    scaler_X.fit(X_train)
#X_train = scaler_X.transform(X_train)
#X_val = scaler_X.transform(X_val)

#with config_context(target_offload="gpu:0"):
#    scaler_y = StandardScaler()
#    scaler_y.fit(y_train)
#y_train = scaler_y.transform(y_train)
#y_val = scaler_y.transform(y_val)

# Fit the linear regression model
params = {"n_jobs": -1, "copy_X": False}
#with config_context(target_offload="gpu:0"):
model = LinearRegression(**params).fit(X_train, y_train)

# Get predictions from model and accuracy
y_predict = model.predict(X_val)
print(f"LinearRegression model.predict produced outputs of type {type(y_predict)}\n")
mse_metric_opt = metrics.mean_squared_error(y_val, y_predict)
print(f"Patched Scikit-learn MSE: {mse_metric_opt}")


