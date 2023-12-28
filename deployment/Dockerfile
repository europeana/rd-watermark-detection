FROM ubuntu:20.04

WORKDIR /code

COPY . /code

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get -y install curl
RUN apt-get install -y git


RUN apt-get install -y python3.9 \
    && ln -s /usr/bin/python3.9 /usr/bin/python3

RUN apt-get install -y python3-setuptools
RUN apt-get install -y python3-pip
RUN apt-get update && apt-get install -y python3-opencv

RUN apt-get install -y automake autotools-dev fuse g++ git libcurl4-openssl-dev libfuse-dev libssl-dev libxml2-dev make pkg-config
RUN apt-get install s3fs

RUN apt-get install -y openjdk-8-jdk

RUN pip install -U pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install -r requirements.txt

RUN pip install pyspark

RUN pip install psycopg2-binary

COPY . /code



