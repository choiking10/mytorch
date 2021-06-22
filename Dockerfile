FROM cupy/cupy:v8.6.0
MAINTAINER Yunho Choi<choiking10@gmail.com>

RUN apt-get update
RUN apt-get install -y git graphviz
#
#COPY requirements.txt /opt
#WORKDIR /opt

# for opencv2
RUN apt install -y libgl1-mesa-glx
RUN pip install opencv-python matplotlib jupyter jupyter_contrib_nbextensions
RUN pip install graphviz
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable hinterland/hinterland

RUN mkdir /workspace
WORKDIR /workspace
