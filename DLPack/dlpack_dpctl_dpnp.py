# DPNP and DPCTL interoperability with DLPack

import os
import dpctl.tensor as dpt
import numpy as np
import dpnp as dp

def dpnp_read_dpctl(device):
    """
    dpnp reading a dpctl array
    """
    dpt_ary = dpt.arange(4)
    dp_ary = dp.from_dlpack(dpt_ary)
    dp_ary[0] = -1.0
    assert 'gpu' or 'Sycl' in str(dpt_ary.device), "dpctl array not on XPU"
    assert 'gpu' or 'Sycl' in str(dp_ary.device), "dpnp array not on XPU"
    np.testing.assert_equal(actual=dpt.asnumpy(dpt_ary), desired=dp.asnumpy(dp_ary))
    return 0

def dpctl_read_dpnp(device):
    """
    dpctl reading a dpnp array
    """
    dp_ary = dp.arange(4)
    dpt_ary = dpt.from_dlpack(dp_ary)
    dpt_ary[0] = -2.0
    assert 'gpu' or 'Sycl' in str(dpt_ary.device), "dpctl array not on XPU"
    assert 'gpu' or 'Sycl' in str(dp_ary.device), "dpnp array not on XPU"
    np.testing.assert_equal(actual=dpt.asnumpy(dpt_ary), desired=dp.asnumpy(dp_ary))
    return 0


## Main
def main():
    print("\n\n")
    device = "xpu"
    
    tests = {}
    try:
        out = dpnp_read_dpctl(device)
        if out == 0:
            tests["dpnp_read_dpctl"] = "Pass"
    except Exception as err:
        print("dpnp_read_dpctl error:")
        print(err,"\n\n")
        tests["dpnp_read_dpctl"] = "Fail with error: " + str(err)

    try:
        out = dpctl_read_dpnp(device)
        if out == 0:
            tests["dpctl_read_dpnp"] = "Pass"
    except Exception as err:
        print("dpctl_read_dpnp error:")
        print(err,"\n\n")
        tests["dpctl_read_dpnp"] = "Fail with error: " + str(err)

    print("\n===============================")
    print("===============================")
    print("Summary of tests:")
    for test in tests:
       print(f"{test}: {tests[test]}","\n")


## Run main
if __name__ == "__main__":
    main()

