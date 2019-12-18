# -*- coding: utf-8 -*-
# @File    : train.py
# @Author  : AaronJny
# @Time    : 2019/12/17
# @Desc    :
import tensorflow as tf
from data import train_db, dev_db
import models
import settings

# 从models文件中导入模型
model = models.my_densenet()
model.summary()

# 配置优化器、损失函数、以及监控指标
model.compile(tf.keras.optimizers.Adam(settings.LEARNING_RATE), loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# 在每个epoch结束后尝试保存模型参数，只有当前参数的val_accuracy比之前保存的更优时，才会覆盖掉之前保存的参数
model_check_point = tf.keras.callbacks.ModelCheckpoint(filepath=settings.MODEL_PATH, monitor='val_accuracy',
                                                       save_best_only=True)
# 使用tf.keras的高级接口进行训练
model.fit_generator(train_db, epochs=settings.TRAIN_EPOCHS, validation_data=dev_db, callbacks=[model_check_point])
