import tensorflow as tf
import numpy as np
# 首先，创建一个TensorFlow常量=>值为2.0
const = tf.constant(2.0, name='const')

# 创建TensorFlow变量b和c
#b = tf.Variable(2.0, name='b')#b值为2.0
#变量b可以接收任意值。TensorFlow中接收值的
#方式为占位符(placeholder)，通过tf.placeholder()创建。
b = tf.placeholder(tf.float32, [None, 1], name='b')
c = tf.Variable(1.0, dtype=tf.float32, name='c')#c值为1.0
# 创建operation
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')
# 1. 定义init operation
init_op = tf.global_variables_initializer()
# session
with tf.Session() as sess:
    # 2. 运行init operation
    sess.run(init_op)
    # 计算
    #a_out = sess.run(a)
    #feed占位符b的值,b的取值为0到9
    a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
    print("Variable a is {}".format(a_out))