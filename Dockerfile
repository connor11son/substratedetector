FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install pandas av opencv-python

WORKDIR /module
COPY  detection /module/detection
COPY multiclass_resnet18.pth /module/multiclass_resnet18.pth
COPY test.mp4 /module/test.mp4