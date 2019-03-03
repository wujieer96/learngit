from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#1.加载数据：
#one_hot=True表示对label进行one-hot编码，比如标签4可以表示为[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]。这是神经网络输出层要求的格式。
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

#2.定义超参数和placeholder
#超参数
learning_rate=0.5
epochs=10
batch_size=100

#placeholder
#输入图片为28x28像素=784
x=tf.placeholder(tf.float32,[None,784])#[None, 784]中的None表示任意值，特别对应tensor数目
#输出为0-9的one_hot编码
y=tf.placeholder(tf.float32,[None,10])

#3.定义参数w和b
#hidden layer=>w,b
W1=tf.Variable(tf.random_normal([784,300],stddev=0.03),name='W1'))
b1=tf.Variable(tf.random_normal([300]),name='b1')
#output layer=>w,b
W2=tf.Variable(tf.random_normal([300,10],stddev=0.03),name='W2'))
b2=tf.Variable(tf.random_normal([10]),name='b2')
