FROM python:3.8

WORKDIR "/root"

# Copy function code

# Add modules
ADD scripts scripts 
ADD configs configs
ADD datasets datasets
ADD models models
ADD pretrained_models pretrained_models
ADD utils utils

# Give execution permission
RUN chmod -R 755 .

# Install dblib
RUN apt update -y && \
    apt install build-essential cmake pkg-config -y
RUN apt update -y && apt install -y gcc g++
RUN pip install cmake
RUN apt install libboost-all-dev -y
RUN apt install make -y
RUN apt install libsm6 libxext6 libxrender1 libfontconfig1 -y
RUN pip install dlib
RUN mkdir -p output
RUN mkdir -p input

# Install the function's dependencies using file requirements.txt
COPY requirements.txt "/root/"
RUN  pip install -r requirements.txt

COPY app.py "/root/"

# Set the CMD to the handler
CMD [ "python", "app.py" ] 