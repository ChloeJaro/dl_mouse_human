FROM ubuntu:22.04

LABEL MAINTAINER="Mohammed E. Amer <mohammed.e.amer@gmail.com>"

ARG DEBIAN_FRONTEND=noninteractive

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

RUN apt update
RUN apt install -y nvidia-driver-470
RUN apt install -y python3-pip
RUN apt install -y openslide-tools
RUN apt install -y ffmpeg libsm6 libxext6

COPY requirements.txt .

RUN pip3 install -r requirements.txt

WORKDIR /usr/src/
USER user

ENTRYPOINT ["./run.sh"]
