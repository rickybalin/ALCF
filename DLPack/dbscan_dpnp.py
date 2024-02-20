# Test interoperability of DPNP and DPCTL with SKLearnX

import numpy as np
import dpctl.tensor as dpt
import dpnp as dp
from daal4py.oneapi import sycl_context
from sklearnex import patch_sklearn, config_context
patch_sklearn()
from sklearn.cluster import DBSCAN

print("")
print("DPNP ...")
X = dp.array([[1., 2.], [2., 2.], [2., 3.],
            [8., 7.], [8., 8.], [25., 80.]], dtype=dp.float32)
print(f"Created DPNP array on device {X.device}")
try: 
    with config_context(target_offload="gpu:0"):
        pred = DBSCAN(eps=3, min_samples=2).fit_predict(X)
    #print(f"Output prediction device {pred.device}")
    print("DPNP interop. pass \n")
except Exception as err:
    print("DPNP interop. error:")
    print(err,"\n")
del X


print("DPCTL ...")
X = dpt.asarray([[1., 2.], [2., 2.], [2., 3.],
            [8., 7.], [8., 8.], [25., 80.]], dtype=dpt.float32)
print(f"Created DPCTL array on device {X.device}")
try: 
    with sycl_context("gpu:0"):
        clustering = DBSCAN(eps=3, min_samples=2).fit(X)
        pred = DBSCAN(eps=3, min_samples=2).fit_predict(X)
    #print(f"Output clustering device {clustering.labels_.device}")
    print("DPCTL interop. pass \n")
except Exception as err:
    print("DPCTL interop. error:")
    print(err,"\n")
del X


