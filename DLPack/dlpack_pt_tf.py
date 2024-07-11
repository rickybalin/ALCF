# PyTorch and TensorFlow interoperability with DLPack

import os
import dpnp as dp
import dpctl
import numpy as np
import torch
import intel_extension_for_pytorch as ipex
import tensorflow as tf
import intel_extension_for_tensorflow as itex

def pt_read_tf(device):
    """
    PyTorch reading a TensorFlow tensor
    """
    with tf.device('/device:XPU:0'):
        tf_ary = tf.range(4)
        dlcapsule = tf.experimental.dlpack.to_dlpack(tf_ary)
        t_ary = torch.from_dlpack(dlcapsule)
        t_ary[0] = -5.0
        print(f"TF: {tf_ary}", f"PT: {t_ary}")
        print(t_ary.device.type)
    assert "XPU" in tf_ary.device, "TensorFlow tensor not on XPU"
    assert t_ary.device.type == "xpu", "PyTorch tensor not on XPU"
    np.testing.assert_equal(actual=tf_ary.numpy(), desired=t_ary.cpu().numpy())
    #print("PyTorch reads a TensorFlow tensor\n")
    return 0

def tf_read_pt(device):
    """
    TensorFlow reading a PyTorch tensor
    """
    t_ary = torch.arange(4).to(device)
    dlcapsule = torch.utils.dlpack.to_dlpack(t_ary)
    with tf.device('/device:XPU:0'):
        tf_ary = tf.experimental.dlpack.from_dlpack(dlcapsule)
    t_ary[0] = -6.0
    #print(f"TF: {tf_ary}", f"PT: {t_ary}")
    assert "XPU" in tf_ary.device, "TensorFlow tensor not on XPU"
    assert t_ary.device.type == "xpu", "PyTorch tensor not on XPU"
    np.testing.assert_equal(actual=tf_ary.numpy(), desired=t_ary.cpu().numpy())
    #print("TesorFlow reads a PyTorch tensor on the XPU\n")
    return 0


## Main
def main():
    print("\n\n")
    device = "xpu"

    tests = {}
    try:
        out = tf_read_pt(device)
        if out == 0:
            tests["tf_read_pt"] = "Pass"
    except Exception as err:
        print("tf_read_pt error:")
        print(err,"\n\n")
        tests["tf_read_pt"] = "Fail with error: " + str(err)

    try:
        out = pt_read_tf(device)
        if out == 0:
            tests["pt_read_tf"] = "Pass"
    except Exception as err:
        print("pt_read_tf error:")
        print(err,"\n\n")
        tests["pt_read_tf"] = "Fail with error: " + str(err)

    print("\n===============================")
    print("===============================")
    print("Summary of tests:")
    for test in tests:
       print(f"{test}: {tests[test]}","\n")


## Run main
if __name__ == "__main__":
    main()

