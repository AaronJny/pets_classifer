# -*- coding: utf-8 -*-
# @File    : app.py
# @Author  : AaronJny
# @Time    : 2019/12/18
# @Desc    :
from flask import Flask
from flask import jsonify
from flask import request, render_template
import tensorflow as tf
from models import my_densenet
import settings

app = Flask(__name__)

# 导入模型
model = my_densenet()
# 加载训练好的参数
model.load_weights(settings.MODEL_PATH)


@app.route('/', methods=['GET'])
def index():
    """
    首页，vue入口
    """
    return render_template('index.html')


@app.route('/api/v1/pets_classify/', methods=['POST'])
def pets_classify():
    """
    宠物图片分类接口，上传一张图片，返回此图片上的宠物是那种类别，概率多少
    """
    # 获取用户上传的图片
    img_str = request.files.get('file').read()
    # 进行数据预处理
    x = tf.image.decode_image(img_str, channels=3)
    x = tf.image.resize(x, (224, 224))
    x = x / 255.
    x = (x - tf.constant(settings.IMAGE_MEAN)) / tf.constant(settings.IMAGE_STD)
    x = tf.reshape(x, (1, 224, 224, 3))
    # 预测
    y_pred = model(x)
    pet_cls_code = tf.argmax(y_pred, axis=1).numpy()[0]
    pet_cls_prob = float(y_pred.numpy()[0][pet_cls_code])
    pet_cls_prob = '{}%'.format(int(pet_cls_prob * 100))
    pet_class = settings.CODE_CLASS_MAP.get(pet_cls_code)
    # 将预测结果组织成json
    res = {
        'code': 0,
        'data': {
            'pet_cls': pet_class,
            'probability': pet_cls_prob,
            'msg': '<br><br><strong style="font-size: 48px;">{}</strong> <span style="font-size: 24px;"'
                   '>概率<strong>{}</strong></span>'.format(pet_class, pet_cls_prob),
        }
    }
    # 返回json数据
    return jsonify(res)


if __name__ == '__main__':
    app.run(port=settings.WEB_PORT)
