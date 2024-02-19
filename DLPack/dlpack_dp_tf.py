# DPNP and TensorFlow interoperability with DLPack

import os
import dpnp as dp
import dpctl
import numpy as np
import tensorflow as tf
import intel_extension_for_tensorflow as itex

def tf_read_dpnp(device):
    """
    Tensorflow reading a dpnp array
    """
    dp_ary = dp.arange(4)
    dlcapsule = dp_ary.__dlpack__()
    print("dpnp dlpack device: ",dp_ary.__dlpack_device__())
    tf_ary = tf.experimental.dlpack.from_dlpack(dlcapsule) # unsupported device type
    dp_ary[0] = -3.0
    #print(f"dpnp: {dp_ary}", f"TF: {tf_ary}")
    assert 'gpu' or 'Sycl' in str(dp_ary.device), "dpnp array not on XPU"
    assert "XPU" in tf_ary.device, "TensorFlow tensor not on XPU"
    np.testing.assert_equal(actual=dp.asnumpy(dp_ary), desired=tf_ary.numpy())
    #print("TensorFlow reads a dpnp array on the XPU\n")
    return 0

def dpnp_read_tf(device):
    """
    dpnp reading a TensorFlow tensor
    """
    with tf.device('/device:XPU:0'):
        tf_ary = tf.range(4)
        dlcapsule = tf.experimental.dlpack.to_dlpack(tf_ary)
    dp_ary = dp.from_dlpack(dlcapsule)
    tf_ary[0] = -4.0
    #print(f"dpnp: {dp_ary}", f"TF: {tf_ary}")
    assert 'gpu' or 'Sycl' in str(dp_ary.device), "dpnp array not on XPU"
    assert "XPU" in tf_ary.device, "TensorFlow tensor not on XPU"
    np.testing.assert_equal(actual=dp.asnumpy(dp_ary), desired=tf_ary.numpy())
    #print("dpnp reads a TensorFlow tensor on the XPU")
    return 0


## Main
def main():
    print("\n\n")
    device = "xpu"

    tests = {}
    try:
        out = tf_read_dpnp(device)
        if out == 0:
            tests["tf_read_dpnp"] = "Pass"
    except Exception as err:
        print("tf_read_dpnp error:")
        print(err,"\n\n")
        tests["tf_read_dpnp"] = "Fail with error: " + str(err)

    try:
        out = dpnp_read_tf(device)
        if out == 0:
            tests["dpnp_read_tf"] = "Pass"
    except Exception as err:
        print("dpnp_read_tf error:")
        print(err,"\n\n")
        tests["dpnp_read_tf"] = "Fail with error: " + str(err)

    print("\n===============================")
    print("===============================")
    print("Summary of tests:")
    for test in tests:
       print(f"{test}: {tests[test]}","\n")


## Run main
if __name__ == "__main__":
    main()

