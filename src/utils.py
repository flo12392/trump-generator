import tensorflow as tf
def print_tensorflow_devices():
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())