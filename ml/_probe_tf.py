import tensorflow as tf
print("before", flush=True)
value = tf.constant([[1.0]], dtype=tf.float32)
print(value.numpy(), flush=True)
print("after", flush=True)
