# 基础镜像
FROM python:3.7-slim
#FROM continuumio/anaconda3
# 工作目录
WORKDIR /app
# 复制代码到镜像
COPY . /app
# 下载预训练的Model
RUN mkdir -p /root/.keras/models \
    && cd /root/.keras/models \
    && wget https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5
#RUN mv /app/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5 /root/.keras/models
# 安装依赖
#RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
RUN pip install -r requirements.txt
# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 内部使用的端口
EXPOSE 5000

ENV PYTHONPATH /app

ENTRYPOINT ["python", "app.py"]