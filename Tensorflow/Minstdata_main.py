from mnistdata import MnistData
import tensorflow as tf

sess=tf.InteractiveSession()

train_image_path='C:\\pythonwork\\project_by_wujie\\Tensorflow\\MNIST_data\\train-images.idx3-ubyte'
train_label_path='C:\\pythonwork\\project_by_wujie\\Tensorflow\\MNIST_data\\train-labels.idx1-ubyte'
test_image_path='C:\\pythonwork\\project_by_wujie\\Tensorflow\\MNIST_data\\t10k-images.idx3-ubyte'
test_label_path='C:\\pythonwork\\project_by_wujie\\Tensorflow\\MNIST_data\\t10k-labels.idx3-ubyte'

epochs=10
batch_size=100
learning_rate=0.2

#创建样本数据的placeholder
x=tf.placeholder(tf.float32,[None,28,28])
#定义权重矩阵和偏置项
#W=tf.Variable(tf.zeros([28*28,10]))
#b=tf.Variable(tf.zeros([10]))
w_1=tf.Variable(tf.truncated_normal([28*28,200],stddev=0.1))
b_1=tf.Variable(tf.zeros([200]))
w_2=tf.Variable(tf.truncated_normal([200,10],stddev=0.1))
b_2=tf.Variable(tf.zeros([10]))

#样本的真实标签
#y_=tf.placeholder((tf.float32,[None,10])
#使用softmax函数将单层网络的输出转换为预测结果
#y=tf.nn.softmax(tf.matmul(tf.reshape(x, [-1, 28*28]), W) + b)

#定义一个两层神经网络模型
y_1=tf.nn.sigmoid(tf.matmul(tf.reshape(x,[-1,28*28]),w_1)+b_1)
y=tf.nn.softmax(tf.matmul(y_1,w_2)+b_2)

#损失函数和优化器
#-tf.reduce_sum(y_*tf.log(y))计算这个batch中每个样本的交叉熵
#reduce_mean方法对一个batch的样本的交叉熵求平均值，作为最终的loss
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_1*tf.log(y),axis=1))
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

#比较预测结果和真实类标
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_1,1))
#计算预测结果和真实类标
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#初始化MnistData类
data=MnistData(train_image_path,train_label_path,test_image_path,test_label_path)
#初始化模型参数
init=tf.global_variables_initializer().run()

#开始训练
for i in range(epochs):
    for j in range(600):
        #获取一个batch的数据
        batch_x,batch_y=data.get_batch(batch_size)
        #优化参数
        train_step.run({x:batch_x,y_1:batch_y})

#对测试集进行预测并计算准确率
print(accuracy.eval({x:data.test_images,y_1:data.test_labels}))      
