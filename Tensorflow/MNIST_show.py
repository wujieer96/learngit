from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as pyplot

#引入 MNIST 数据集
mnist = input_data.read_data_sets("C:\\pythonwork\\project_by_wujie\\Tensorflow\\MNIST_data", one_hot=False)

#选取训练集中的第 1 个图像的矩阵
mnist_one=mnist.train.images[1]

#输出图片的维度，结果是：(784,)
print(mnist_one.shape)

#因为原始的数据是长度是 784 向量，需要转换成 28*28 的矩阵。
mnist_one_image=mnist_one.reshape((28,28))

#输出矩阵的维度
print(mnist_one_image.shape)

#使用 matplotlib 输出为图片
pyplot.imshow(mnist_one_image)

pyplot.show()
