# -*- coding: utf-8 -*-
# @File    : models.py
# @Author  : AaronJny
# @Time    : 2019/12/16
# @Desc    :
import tensorflow as tf
import settings


def my_densenet():
    """
    创建并返回一个基于densenet的Model对象
    """
    # 获取densenet网络，使用在imagenet上训练的参数值，移除头部的全连接网络，池化层使用max_pooling
    densenet = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', pooling='max')
    # 冻结预训练的参数，在之后的模型训练中不会改变它们
    densenet.trainable = False
    # 构建模型
    model = tf.keras.Sequential([
        # 输入层，shape为(None,224,224,3)
        tf.keras.layers.Input((224, 224, 3)),
        # 输入到DenseNet121中
        densenet,
        # 将DenseNet121的输出展平，以作为全连接层的输入
        tf.keras.layers.Flatten(),
        # 添加BN层
        tf.keras.layers.BatchNormalization(),
        # 随机失活
        tf.keras.layers.Dropout(0.5),
        # 第一个全连接层，激活函数relu
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        # BN层
        tf.keras.layers.BatchNormalization(),
        # 随机失活
        tf.keras.layers.Dropout(0.5),
        # 第二个全连接层，激活函数relu
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        # BN层
        tf.keras.layers.BatchNormalization(),
        # 输出层，为了保证输出结果的稳定，这里就不添加Dropout层了
        tf.keras.layers.Dense(settings.CLASS_NUM, activation=tf.nn.softmax)
    ])

    return model


if __name__ == '__main__':
    model = my_densenet()
    model.summary()
