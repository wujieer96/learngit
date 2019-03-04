import numpy as np 
import struct
import random

#首先定义一个MnistData类来管理数据：
class MnistData:
    def __init__(self,train_image_path,train_label_path,test_image_path,test_label_path):
        #训练集和测试集的文件路径
        self.train_image_path=train_image_path
        self.train_label_path=train_label_path
        self.test_image_path=test_image_path
        self.test_label_path=test_label_path

        #获取训练集和测试集数据
        #get_data()方法，参数为0获取训练集数据，参数为1获取测试集
        self.train_images,self.train_labels=self.get_data(0)
        self.test_images,self.test_labels=self.get_data(0)

        #定义两个辅助变量，用来判断一个回合的训练是否完成
        self.num_of_batch=0
        self.got_batch=0

        #get_data实现了Mnist数据集的读取以及数据的预处理
    def get_data(self,data_type):
        if data_type==0:#获取训练集数据
            image_path=self.train_image_path
            label_path=self.train_label_path
        else:#获取测试集数据
            image_path=self.test_image_path
            label_path=self.test_label_path
            
        with open(image_path,'rb')as file1:
            image_file=file1.read()
        with open(label_path,'rb')as file2:
            label_file=file2.read()
            
        label_index=0
        image_index=0
        labels=[]
        images=[]

        #读取训练集图像数据文件的文件信息
        #struct模块处理二进制文件，uppack_from函数用来解包二进制文件
        #‘>’代表二进制文件是以大端法存储，‘||||’代表四个int类型的长度，一个int类型占四个字节，指定读取16字节的内容
        magic,num_of_datasets,rows,columns=struct.unpack_from('>||||',image_file,image_index)
        image_index+=struct.calcsize('>||||')#image_index是偏移量

        for i in range(num_of_datasets):
            #读取784个unsigned byte,即一副图像的所有像素值
            temp=struct.unpack_from('>784B',image_file,image_index)
            #将读取的像素数据转换成28*28的矩阵
            temp=np.reshape(temp,(28,28))
            #归一化处理
            temp=temp/255
            images.append(temp)
            image_index+=struct.calcsize('>784B')#每次增加784B
        
        #跳过描述信息
        label_index+=struct.calcsize('>||')
        labels=struct.unpack_from('>'+str(num_of_datasets)+'B',label_file,label_index)

        #one_hot
        labels=np.eye(10)[np.array(labels)]

        return images,labels
    def get_batch(self,batch_size):
        #刚开始训练或当一轮训练结束之后，打乱数据集数据的顺序
        if self.got_batch==self.num_of_batch:
            train_list=list(zip(self.train_images,self.train_labels))
            random.shuffle(train_list)
            self.train_images,self.train_labels=zip(*train_list)
        
            #重置两个辅助变量
            self.num_of_batch=60000/batch_size
            self.got_batch=0
        #获取一个batch size的训练数据
        train_images=self.train_images[self.got_batch*batch_size:(self.got_batch+1)*batch_size]
        train_labels=self.train_labels[self.got_batch*batch_size:(self.got_batch+1)*batch_size]
        self.got_batch+=1

        return train_images,train_labels
    

                
