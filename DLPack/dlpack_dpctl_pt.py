# DPNP and PyTorch interoperability with DLPack

import os
import dpctl.tensor as dpt
import numpy as np
import torch
import intel_extension_for_pytorch as ipex

def pt_read_dpctl(device):
    """
    PyTorch reading a dpctl array
    """
    dpt_ary = dpt.arange(4)
    t_ary = torch.from_dlpack(dpt_ary)
    t_ary[0] = -1.0
    #print(f"dpctl: {dpt_ary}", f"PT: {t_ary}")
    assert t_ary.device.type == "xpu", "PyTorch tensor not on XPU"
    assert 'gpu' or 'Sycl' in str(dpt_ary.device), "dpctl array not on XPU"
    np.testing.assert_equal(actual=dpt.asnumpy(dpt_ary), desired=t_ary.cpu().numpy())
    #print("PyTorch reads a dpctl array on the XPU\n")
    return 0

def dpctl_read_pt(device):
    """
    dpctl reading a Pytorch tensor
    """
    t_ary = torch.arange(4).to(device)
    dpt_ary = dpt.from_dlpack(t_ary)
    t_ary[0] = -2.0
    #print(f"dpctl: {dpt_ary}", f"PT: {t_ary}")
    assert 'gpu' or 'Sycl' in str(dpt_ary.device), "dpctl array not on XPU"
    assert t_ary.device.type == "xpu", "PyTorch tensor not on XPU"
    np.testing.assert_equal(actual=dpt.asnumpy(dpt_ary), desired=t_ary.cpu().numpy())
    #print("dpctl reads a PyTorch tensor on the XPU\n")
    return 0


## Main
def main():
    print("\n\n")
    device = "xpu"
    
    tests = {}
    try:
        out = pt_read_dpctl(device)
        if out == 0:
            tests["pt_read_dpctl"] = "Pass"
    except Exception as err:
        print("pt_read_dpctl error:")
        print(err,"\n\n")
        tests["pt_read_dpctl"] = "Fail with error: " + str(err)

    try:
        out = dpctl_read_pt(device)
        if out == 0:
            tests["dpctl_read_pt"] = "Pass"
    except Exception as err:
        print("dpctl_read_pt error:")
        print(err,"\n\n")
        tests["dpctl_read_pt"] = "Fail with error: " + str(err)

    print("\n===============================")
    print("===============================")
    print("Summary of tests:")
    for test in tests:
       print(f"{test}: {tests[test]}","\n")


## Run main
if __name__ == "__main__":
    main()

