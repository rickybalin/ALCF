# Test interoperability of DPNP and DPCTL with SKLearnX

import numpy as np
import dpctl.tensor as dpt
import dpnp as dp
from daal4py.oneapi import sycl_context
from sklearnex import patch_sklearn, config_context
patch_sklearn()
from sklearn.model_selection import train_test_split

print("")
print("DPNP ...")
X = dp.random.rand(10,6)
y = dp.random.rand(10,1)
print(f"Created DPNP arrays on device {X.device} and {y.device}")
try: 
    with config_context(target_offload="gpu:0"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print(f"Split arrays on device {X_train.device}")
    print("DPNP interop. pass \n")
except Exception as err:
    print("DPNP interop. error:")
    print(err,"\n")
del X,y


print("DPCTL ...")
X = np.random.rand(10,6)
y = np.random.rand(10,1)
X = dpt.from_numpy(X)
y = dpt.from_numpy(y)
print(f"Created DPCTL arrays on device {X.device} and {y.device}")
try: 
    #with sycl_context("level_zero:gpu:0"):
    with config_context(target_offload="gpu:0"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print(f"Split arrays on device {X_train.device}")
    print("DPCTL interop. pass \n")
except Exception as err:
    print("DPCTL interop. error:")
    print(err,"\n")
del X,y


