# DPNP and PyTorch interoperability with DLPack

import os
import dpnp as dp
import dpctl
import numpy as np
import torch
import intel_extension_for_pytorch as ipex

def pt_read_dpnp(device):
    """
    PyTorch reading a dpnp array
    """
    dp_ary = dp.arange(4)
    t_ary = torch.from_dlpack(dp_ary)
    t_ary[0] = -1.0
    #print(f"dpnp: {dp_ary}", f"PT: {t_ary}")
    assert t_ary.device.type == "xpu", "PyTorch tensor not on XPU"
    assert 'gpu' or 'Sycl' in str(dp_ary.device), "dpnp array not on XPU"
    np.testing.assert_equal(actual=dp.asnumpy(dp_ary), desired=t_ary.cpu().numpy())
    #print("PyTorch reads a dpnp array on the XPU\n")
    return 0

def dpnp_read_pt(device):
    """
    dpnp reading a Pytorch tensor
    """
    t_ary = torch.arange(4).to(device)
    dp_ary = dp.from_dlpack(t_ary)
    t_ary[0] = -2.0
    #print(f"dpnp: {dp_ary}", f"PT: {t_ary}")
    assert 'gpu' or 'Sycl' in str(dp_ary.device), "dpnp array not on XPU"
    assert t_ary.device.type == "xpu", "PyTorch tensor not on XPU"
    np.testing.assert_equal(actual=dp.asnumpy(dp_ary), desired=t_ary.cpu().numpy())
    #print("dpnp reads a PyTorch tensor on the XPU\n")
    return 0


## Main
def main():
    print("\n\n")
    device = "xpu"
    
    tests = {}
    try:
        out = pt_read_dpnp(device)
        if out == 0:
            tests["pt_read_dpnp"] = "Pass"
    except Exception as err:
        print("pt_read_dpnp error:")
        print(err,"\n\n")
        tests["pt_read_dpnp"] = "Fail with error: " + str(err)

    try:
        out = dpnp_read_pt(device)
        if out == 0:
            tests["dpnp_read_pt"] = "Pass"
    except Exception as err:
        print("dpnp_read_pt error:")
        print(err,"\n\n")
        tests["dpnp_read_pt"] = "Fail with error: " + str(err)

    print("\n===============================")
    print("===============================")
    print("Summary of tests:")
    for test in tests:
       print(f"{test}: {tests[test]}","\n")


## Run main
if __name__ == "__main__":
    main()

