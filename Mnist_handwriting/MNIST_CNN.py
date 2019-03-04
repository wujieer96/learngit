import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

mnist_data=input_data.read_data_sets("C:\\pythonwork\\project_by_wujie\\Tensorflow\\MNIST_data",one_hot=True)
x = tf.placeholder(tf.float32,[None,784])#训练数据， 28x28=784，代表把图片转换为长度为784的向量
y = tf.placeholder(tf.float32,[None,10])#标签，表示10个不同的类标

weights = {
'wc1':tf.Variable(tf.random_normal([5,5,1,32])),
'wc2':tf.Variable(tf.random_normal([5,5,32,64])),
'wd1':tf.Variable(tf.random_normal([7*7*64,1024])),
'out':tf.Variable(tf.random_normal([1024,10]))  
}
biases = {
'bc1': tf.Variable(tf.random_normal([32])), 
'bc2': tf.Variable(tf.random_normal([64])),  
'bd1': tf.Variable(tf.random_normal([1024])),
'out': tf.Variable(tf.random_normal([10])),
}

#定义一个函数，用于构建卷积层
def conv2d(x,W,b,strides=1):
    #Conv2d wrapper,with bias and relu activation
    x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x=tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

#定义一个函数，用于构建池化层
def maxpool2d(x,k=2):
    #MaxPool2D wrapper
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

#超参数定义：
#训练参数
learning_rate=0.001
num_steps=200
batch_size=128
display_step=10
batch_num=mnist_data.train.num_examples//batch_size
#网络参数
#MNIST 数据维度
num_input=784
#MNIST 列标数量
num_classes=10
#神经元保留率
dropout=0.75

#构建网络
def conv_net(x,weights,biases,dropout):
    x=tf.reshape(x,shape=[-1,28,28,1])
    #第一层卷积
    conv1=conv2d(x,weights['wc1'],biases['bc1'])
    #第二层池化
    conv1=maxpool2d(conv1,k=2)
    #第三层卷积
    conv2=conv2d(conv1,weights['wc2'],biases['bc2'])
    #第四层池化
    conv2=maxpool2d(conv2,k=2)

    #全连接层
    fc1=tf.reshape(conv2,[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1=tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1=tf.nn.relu(fc1)
    #丢弃部分神经元
    fc1=tf.nn.dropout(fc1,dropout)

    #输出层，输出最后的结果
    out=tf.add(tf.matmul(fc1,weights['out']),biases['out'])
    return out

#调用神经网络
#softmax层
logits=conv_net(x,weights,biases,0.75)
prediction=tf.nn.softmax(logits)

#定义损失函数
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#定义优化函数
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
#确定优化目标
train_op=optimizer.minimize(loss_op)

#获得预测正确的结果
correct_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
#准确率
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
""" sess=tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in range(15000):
    batch_x,batch_y = mnist_data.train.next_batch(batch_size)
    if i%100==0:
        train_acc=accuracy.eval(feed_dict={x:batch_x,y:batch_y})
        print('step',i,'Training accuracy',train_acc)
        train_op.run(feed_dict={x:batch_x,y:batch_y})
test_acc=accuracy.eval(feed_dict={x:mnist_data.test.images,y:mnist_data.test.labels})
print("test accuracy",test_acc)
"""
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(num_steps+1):
        for batch in range(batch_num):
            batch_x,batch_y = mnist_data.train.next_batch(batch_size)
            sess.run(train_op,feed_dict={x:batch_x,y:batch_y})#更新模型权重
            #print(train_step)
            acc = sess.run(accuracy,feed_dict={x:mnist_data.test.images,y:mnist_data.test.labels})
        if step%10==0:
            print("Step " + str(step) + ",Training Accuracy "+ "{:.3f}" + str(acc))
    print("Finished!")
