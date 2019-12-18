# -*- coding: utf-8 -*-
# @File    : eval.py
# @Author  : AaronJny
# @Time    : 2019/12/17
# @Desc    :
import tensorflow as tf
from data import dev_db, test_db
from models import my_densenet
import settings

# 创建模型
model = my_densenet()
# 加载参数
model.load_weights(settings.MODEL_PATH)
# 因为想用tf.keras的高级接口做验证，所以还是需要编译模型
model.compile(tf.keras.optimizers.Adam(settings.LEARNING_RATE), loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
# 验证集accuracy
print('dev', model.evaluate(dev_db))
# 测试集accuracy
print('test', model.evaluate(test_db))

# 查看识别错误的数据
for x, y in test_db:
    y_pred = model(x)
    y_pred = tf.argmax(y_pred, axis=1).numpy()
    y_true = tf.argmax(y, axis=1).numpy()
    batch_size = y_pred.shape[0]
    for i in range(batch_size):
        if y_pred[i] != y_true[i]:
            print('{} 被错误识别成 {}!'.format(settings.CODE_CLASS_MAP[y_true[i]], settings.CODE_CLASS_MAP[y_pred[i]]))
