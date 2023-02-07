# syntax=docker/dockerfile:1

FROM ubuntu:latest
RUN set -xe \
    && apt-get update -y\
    && apt-get install -y python3-pip \
    && apt install -y python3.8
RUN pip install --upgrade pip
WORKDIR /app
COPY requirements.txt /app/.
RUN pip3 install -r requirements.txt
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
# Install pangolin
RUN git clone https://github.com/uoip/pangolin.git
RUN cd pangolin
RUN mkdir build
RUN cd build
RUN cmake ..
RUN make -j8
RUN cd ..
RUN python setup.py install

COPY . app/.
RUN pip3 install -e .
ENV QT_DEBUG_PLUGINS=1
CMD [ "python3", "main.py"]
