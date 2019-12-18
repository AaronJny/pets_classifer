# -*- coding: utf-8 -*-
# @File    : data.py
# @Author  : AaronJny
# @Time    : 2019/12/16
# @Desc    :
import os
import random
import tensorflow as tf
import settings

# 每个类别选取的图片数量
samples_per_class = settings.SAMPLES_PER_CLASS
# 图片根目录
images_root = settings.IMAGES_ROOT
# 类别->编码的映射
class_code_map = settings.CLASS_CODE_MAP

# 我们准备使用经典网络在imagenet数据集上的与训练权重，所以归一化时也要使用imagenet的平均值和标准差
image_mean = tf.constant(settings.IMAGE_MEAN)
image_std = tf.constant(settings.IMAGE_STD)


def normalization(x):
    """
    对输入图片x进行归一化，返回归一化的值
    """
    return (x - image_mean) / image_std


def train_preprocess(x, y):
    """
    对训练数据进行预处理。
    注意，这里的参数x是图片的路径，不是图片本身；y是图片的标签值
    """
    # 读取图片
    x = tf.io.read_file(x)
    # 解码成张量
    x = tf.image.decode_jpeg(x, channels=3)
    # 将图片缩放到[244,244]，比输入[224,224]稍大一些，方便后面数据增强
    x = tf.image.resize(x, [244, 244])
    # 随机决定是否左右镜像
    if random.choice([0, 1]):
        x = tf.image.random_flip_left_right(x)
    # 随机从x中剪裁出(224,224,3)大小的图片
    x = tf.image.random_crop(x, [224, 224, 3])
    # 读完上面的代码可以发现，这里的数据增强并不增加图片数量，一张图片经过变换后，
    # 仍然只是一张图片，跟我们前面说的增加图片数量的逻辑不太一样。
    # 这么做主要是应对我们的数据集里可能会存在相同图片的情况。

    # 将图片的像素值缩放到[0,1]之间
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 归一化
    x = normalization(x)

    # 将标签转成one-hot形式
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, settings.CLASS_NUM)

    return x, y


def dev_preprocess(x, y):
    """
    对验证集和测试集进行数据预处理的方法。
    和train_preprocess的主要区别在于，不进行数据增强，以保证验证结果的稳定性。
    """
    # 读取并缩放图片
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.image.resize(x, [224, 224])
    # 归一化
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalization(x)
    # 将标签转成one-hot形式
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, settings.CLASS_NUM)

    return x, y


# (图片路径,标签)的列表
image_path_and_labels = []
# 排序，保证每次拿到的顺序都一样
sub_images_dir_list = sorted(list(os.listdir(images_root)))
# 遍历每一个子目录
for sub_images_dir in sub_images_dir_list:
    sub_path = os.path.join(images_root, sub_images_dir)
    # 如果给定路径是文件夹,并且这个类别参与训练
    if os.path.isdir(sub_path) and sub_images_dir in settings.CLASSES:
        # 获取当前类别的编码
        current_label = class_code_map.get(sub_images_dir)
        # 获取子目录下的全部图片名称
        images = sorted(list(os.listdir(sub_path)))
        # 随机打乱(排序和置随机数种子都是为了保证每次的结果都一样)
        random.seed(settings.RANDOM_SEED)
        random.shuffle(images)
        # 保留前settings.SAMPLES_PER_CLASS个
        images = images[:samples_per_class]
        # 构建(x,y)对
        for image_name in images:
            abs_image_path = os.path.join(sub_path, image_name)
            image_path_and_labels.append((abs_image_path, current_label))
# 计算各数据集样例数
total_samples = len(image_path_and_labels)  # 总样例数
train_samples = int(total_samples * settings.TRAIN_DATASET)  # 训练集样例数
dev_samples = int(total_samples * settings.DEV_DATASET)  # 开发集样例数
test_samples = total_samples - train_samples - dev_samples  # 测试集样例数
# 打乱数据集
random.seed(settings.RANDOM_SEED)
random.shuffle(image_path_and_labels)
# 将图片数据和标签数据分开，此时它们仍是一一对应的
x_data = tf.constant([img for img, label in image_path_and_labels])
y_data = tf.constant([label for img, label in image_path_and_labels])
# 开始划分数据集
# 训练集
train_db = tf.data.Dataset.from_tensor_slices((x_data[:train_samples], y_data[:train_samples]))
# 打乱顺序，数据预处理，设置批大小
train_db = train_db.shuffle(10000).map(train_preprocess).batch(settings.BATCH_SIZE)
# 开发集(验证集)
dev_db = tf.data.Dataset.from_tensor_slices(
    (x_data[train_samples:train_samples + dev_samples], y_data[train_samples:train_samples + dev_samples]))
# 数据预处理，设置批大小
dev_db = dev_db.map(dev_preprocess).batch(settings.BATCH_SIZE)
# 测试集
test_db = tf.data.Dataset.from_tensor_slices(
    (x_data[train_samples + dev_samples:], y_data[train_samples + dev_samples:]))
# 数据预处理，设置批大小
test_db = test_db.map(dev_preprocess).batch(settings.BATCH_SIZE)
