  FROM tensorflow/tensorflow:1.13.0-devel

  MAINTAINER Jerome WAX "xblaster@lo2k.net"

  WORKDIR /tensorflow

  ADD src .

  RUN git pull
  CMD cd /tensorflow && ./train.sh
