# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster
ENV DEBIAN_FRONTEND noninteractive
WORKDIR /.
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN apt-get update -y
RUN pip3 install -r requirements.txt
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
COPY . .
RUN pip3 install -e .
CMD [ "python3", "main.py"]
