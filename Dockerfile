FROM tensorflow/tensorflow:1.8.0-py3

RUN apt-get update && apt-get install -y --no-install-recommends nginx curl

# Download TensorFlow Serving
# https://www.tensorflow.org/serving/setup#installing_the_modelserver
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list
RUN curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
RUN apt-get update && apt-get install tensorflow-model-server

ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY src /opt/ml/code
COPY train /opt/ml/code
COPY requirements.txt /opt/ml/code
WORKDIR /opt/ml/code
RUN pip install -r requirements.txt