FROM tensorflow/tensorflow:2.3.0-gpu-jupyter

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt