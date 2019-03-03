import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#从默认的地方下载MNIST数据集，下载下来的数据集会以压缩包的形式存到指定目录，如下图所示。这些数据分别代表了训练集、训练集标签、测试集、测试集标签
mnist_data=input_data.read_data_sets("C:\\pythonwork\\project_by_wujie\\Tensorflow\\MNIST_data",one_hot=True)
batch_size=100#一次性传入神经网络进行训练的批次大小，100张图片
batch_num=mnist_data.train.num_examples//batch_size#计算出训练的次数

x = tf.placeholder(tf.float32,[None,784])#训练数据， 28x28=784，代表把图片转换为长度为784的向量
y = tf.placeholder(tf.float32,[None,10])#标签，表示10个不同的类标

#程序4：
#定义各层的权重w和偏执b
weights = {
'hidden_1': tf.Variable(tf.random_normal([784, 256])),#隐藏层权重，784x256的权重矩阵
'out': tf.Variable(tf.random_normal([256, 10]))#输出层权重
}
biases = {
'b1': tf.Variable(tf.random_normal([256])),
'out': tf.Variable(tf.random_normal([10]))
}

#搭建一个含有一个隐藏层结构的神经网络,设置其每层的w和b
#该隐藏层含有256个神经元。接着我们就可以开始搭建每一层神经网络了
""" 
def neural_network(x):
    hidden_layer_1 = tf.add(tf.matmul(x, weights['hidden_1']), biases['b1'])#wx+b
    out_layer = tf.matmul(hidden_layer_1, weights['out']) + biases['out']
    return out_layer 
"""
#这里我们定义了一个含有一个隐藏层神经网络的函数neural_network，函数的返回值是输出层的输出结果
""" 
考虑到训练过程可能会发生过拟合现象，所以我们可以从防止过拟合的角度出发，提高模型的准确率。
我们可以采用增加数据量或是增加正则化项的方式，来缓解过拟合。这里，我们为大家介绍dropout
的方式是如何缓解过拟合的。
Dropout是在每次神经网络的训练过程中，使得部分神经元工作而另外一部分神经元不工作。而测试
的时候激活所有神经元，测试及验证中：每个神经元都要参加运算，但其输出要乘以概率p，用所有的神经元进行测试。这样便可以有效的缓解过拟合，提高模型的准确率。
"""
def neural_network(x):
    hidden_layer_1 = tf.add(tf.matmul(x, weights['hidden_1']), biases['b1'])
    L1 = tf.nn.tanh(hidden_layer_1)#双曲线切线激活函数
    dropout1 = tf.nn.dropout(L1,0.5)#0.5表示神经元被选中的概率
    #我们在隐藏层后接了dropout，随机关掉50%的神经元
    out_layer = tf.matmul(dropout1, weights['out']) + biases['out']
    return out_layer


#调用神经网络
result = neural_network(x)
 #预测类别
prediction = tf.nn.softmax(result)#使用softmax函数对结果进行预测
#接下来可以从以下几方面对模型进行改良和优化，以提高模型的准确率。
""" 
首先，在计算损失函数时，可以选择交叉熵损失函数来代替平方差损失
函数，通常在Tensorflow深度学习中，softmax_cross_entropy_with_logits
函数会和softmax函数搭配使用，是因为交叉熵在面对多分类问题时，迭代过程
中权值和偏置值的调整更加合理，模型收敛的速度更加快，训练的的效果也更加好
 """
#改变损失函数
#采用交叉熵损失函数
#loss = tf.reduce_mean(tf.square(y-prediction))#选择平方差损失函数再求其平均值计算出loss

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

#Adam优化算法  使用adam优化算法代替随机梯度下降法，因为它的收敛速度要比随机梯度下降更快，
#这样也能够使准确率有所提高
#minimize(loss)梯度减少最快，也就是更加容易找到函数loss的最小值
#train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)#最小化损失函数时，可以通过梯度下降法来一步步的迭代求解，得到最小化的损失函数，和模型参数值

#平方差损失函数
#loss = tf.reduce_mean(tf.square(y-prediction))#选择平方差损失函数计算出loss
 #梯度下降法loss   使用梯度下降法的优化方法对loss进行最小化（梯度下降法的学习率设置为0.2）。
#train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
 #预测类标  再使用equal函数与正确的类标进行比较，返回一个bool值，代表预测正确或错误的类标
correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
 #计算准确率  使用cast函数把bool类型的预测结果转换为float类型（True转换为1，False转换为0），
 # 并对所有预测结果统计求平均值，算出最后的准确率
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
 #初始化变量，对程序中的所有变量进行初始化
init = tf.global_variables_initializer()

step_num=400#定义迭代的周期数
#对于每一轮的迭代过程，我们用不同批次的图片进行训练，每次训练100张图片，每次训练的图片数据
#和对应的标签分别保存在 batch_x、batch_y中，接着再用run方法执行这个迭代过程，
#并使用feed_dict的字典结构填充每次的训练数据。循环往复上述过程，直到最后一轮的训练结束。
with tf.Session() as sess:
    sess.run(init)
    for step in range(step_num+1):
        for batch in range(batch_num):
            batch_x,batch_y = mnist_data.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_x,y:batch_y})#更新模型权重
            #print(train_step)
            acc = sess.run(accuracy,feed_dict={x:mnist_data.test.images,y:mnist_data.test.labels})
        print("Step " + str(step) + ",Training Accuracy "+ "{:.3f}" + str(acc))
    print("Finished!")



