import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 创建数据训练数据集,
#x_data = np.linspace(-1, 1, 500).reshape(500, 1)
#noise = np.random.normal(0, 0.05, [500, 1])  # 制作噪音
#y_data = np.square(x_data) + 0.5 + noise
x_data = np.linspace(1, 60, 60)[:, np.newaxis] #-1到1等分300份形成的二维矩阵
print(x_data)
noise = np.random.normal(0, 0.05, x_data.shape) #噪音，形状同x_data在0-0.05符合正态分布的小数
y_data = np.square(x_data)+noise #x_data平方，减0.05，再加噪音值


# 创建占位符用于minibatch的梯度下降训练,建议数据类型使用tf.float32、tf.float64等浮点型数据
x_in = tf.placeholder(tf.float32, [None, 1])
y_in = tf.placeholder(tf.float32, [None, 1])


# 定义一个添加层的函数
def add_layer(input_, in_size, out_size, activation_funtion=None):
    '''
    :param input_: 输入的tensor
    :param in_size: 输入的维度，即上一层的神经元个数
    :param out_size: 输出的维度，即当前层的神经元个数即当前层的
    :param activation_funtion: 激活函数
    :return: 返回一个tensor
    '''
    weight = tf.Variable(tf.random_normal([in_size, out_size]))  # 权重，随机的in_size*out_size大小的权重矩阵
    biase = tf.Variable(tf.zeros([1, out_size]) + 0.01)  # 偏置，1*out_size大小的0.01矩阵，不用0矩阵避免计算出错
    if not activation_funtion:  # 根据是否有激活函数决定输出
        output = tf.matmul(input_, weight) + biase
    else:
        output = activation_funtion(tf.matmul(input_, weight) + biase)
    return output


# 定义隐藏层,输入为原始数据，特征为1，所以输入为1个神经元，输出为4个神经元
layer1 = add_layer(x_in, 1, 4, tf.nn.relu)

# 定义输出层,输入为layer1返回的tensor，输入为4个神经元，输出为1个神经元，激活函数为ReLU
predict = add_layer(layer1, 4, 1)

# 定义损失函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_in - predict), axis=[1]))  # tf.reduce_sum的axis=[1]表示按列求和

# 定义训练的优化方式为梯度下降
train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)  # 学习率为0.1

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 训练1000次
    for step in range(10000):
        # 执行训练,因为有占位符所以要传入字典，占位符的好处是可以用来做minibatch训练，这里数据量小，直接传入全部数据来训练
        sess.run(train, feed_dict={x_in: x_data, y_in: y_data})
        # 每50步输出一次loss
        if step % 49 == 0:
            print('loss:', sess.run(loss, feed_dict={x_in: x_data, y_in: y_data}))

    # 最后画出实际的散点图与拟合的折线图进行对比
    predict_value = sess.run(predict, feed_dict={x_in: x_data})  # 先要获得预测值
    plt.figure()
    plt.scatter(x_data, y_data, c='r', marker='o')
    plt.plot(x_data, predict_value, '--', lw=2, c='b')
    plt.show()