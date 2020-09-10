import tensorflow as tf
import math


def log2(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
  return numerator / denominator

if __name__ == '__main__':
    # dev_cnt = 1
    dev_cnt = 0

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25),
        device_count = {'GPU': dev_cnt}
    )
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        a_c = 2 #tf.constant(2)
        b_c = 3 #tf.constant(3)
    
        t1 = [1., 2.] # tf.constant([1, 2])
        t2 = [4., 3.] # tf.constant([4, 3])

        a = tf.placeholder(tf.int32, shape=(), name="a")
        b = tf.placeholder(tf.int32, shape=(), name="b")
        c = tf.placeholder(tf.float32, shape=(2,), name="c")
        d = tf.placeholder(tf.float32, shape=(2,), name="d")

        add = tf.add(a, b)
        mul = d * log2(d)
 
        add1 = sess.run([add], feed_dict={a: a_c, b: b_c})
        print(add1)
    
        mul1 = sess.run([mul], feed_dict={c: t1, d: t2})
        print(mul1)
