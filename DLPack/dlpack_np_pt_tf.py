import numpy as np
import torch
import tensorflow as tf

print("\n\n")

# PyTorch reading a numpy array
np_ary = np.arange(4)
t_ary = torch.from_dlpack(np_ary)
t_ary[0] = -1.0
print(f"Numpy: {np_ary}", f"PT: {t_ary}")
np.testing.assert_equal(actual=np_ary, desired=t_ary.numpy())
print("PyTorch reads a Numpy array\n")
del np_ary, t_ary

# Numpy reading a Pytorch tensor
t_ary = torch.arange(4)
np_ary = np.from_dlpack(t_ary)
t_ary[0] = -2.0
print(f"Numpy: {np_ary}", f"PT: {t_ary}")
np.testing.assert_equal(actual=np_ary, desired=t_ary.numpy())
print("Numpy reads a PyTorch tensor\n")
del np_ary, t_ary

# Tensorflow reading a numpy array
np_ary = np.arange(4)
dlcapsule = np_ary.__dlpack__()
tf_ary = tf.experimental.dlpack.from_dlpack(dlcapsule)
np_ary[0] = -3.0
print(f"Numpy: {np_ary}", f"TF: {tf_ary}")
np.testing.assert_equal(actual=np_ary, desired=tf_ary.numpy())
print("TensorFlow reads a Numpy array\n")
del np_ary, tf_ary, dlcapsule

# Numpy reading a TensorFlow tensor
#tf_ary = tf.range(4)
#dlcapsule = tf.experimental.dlpack.to_dlpack(tf_ary)
#np_ary = np.from_dlpack(dlcapsule)
#tf_ary[0] = -4.0
#print(f"Numpy: {np_ary}", f"TF: {tf_ary}")
#np.testing.assert_equal(actual=np_ary, desired=tf_ary.numpy())
#print("Numpy reads a TensorFlow tensor")
#del np_ary, tf_ary, dlcapsule

# PyTorch reading a TensorFlow tensor
tf_ary = tf.range(4)
dlcapsule = tf.experimental.dlpack.to_dlpack(tf_ary)
t_ary = torch.from_dlpack(dlcapsule)
t_ary[0] = -5.0
print(f"TF: {tf_ary}", f"PT: {t_ary}")
np.testing.assert_equal(actual=tf_ary.numpy(), desired=t_ary.numpy())
print("PyTorch reads a TensorFlow tensor\n")
del t_ary, tf_ary, dlcapsule

# TensorFlow reading a PyTorch tensor
t_ary = torch.arange(4)
dlcapsule = torch.utils.dlpack.to_dlpack(t_ary)
tf_ary = tf.experimental.dlpack.from_dlpack(dlcapsule)
t_ary[0] = -6.0
print(f"TF: {tf_ary}", f"PT: {t_ary}")
np.testing.assert_equal(actual=tf_ary.numpy(), desired=t_ary.numpy())
print("TesorFlow reads a PyTorch tensor\n")
del t_ary, tf_ary, dlcapsule

