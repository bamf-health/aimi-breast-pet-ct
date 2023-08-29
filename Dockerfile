# FROM pytorch/pytorch:latest
# Specify the base image for the environment
# FROM ubuntu:20.04
# FROM --platform=linux/amd64 pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime AS builder
# FROM nvcr.io/nvidia/pytorch:22.01-py3
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04


# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281 
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install system utilities and CUDA related dependencies
RUN apt update && apt install -y --no-install-recommends \
    dcm2niix \
    wget \
    vim \
    p7zip \
    p7zip-full \
    zip \
    unzip \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# # FOllowing are removed
# RUN apt update && apt install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0

# Install python tools needed for nnUNet inference
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    nnunet \
    pydicom \
    SimpleITK \
    dcm2niix \
    pyyaml \
    scikit-build \
    TotalSegmentator \
    pynrrd

# Set work dirs
RUN mkdir /app /app/data /app/data/input_data /app/data/output_data
WORKDIR /app

# Pull weights into the container
ENV WEIGHTS_DIR=/root/.nnunet/nnUNet_models/nnUNet/
RUN mkdir -p $WEIGHTS_DIR
ENV TASK_NAME=Task762_PET_CT_Breast
ENV WEIGHTS_FN=$TASK_NAME.zip
ENV WEIGHTS_URL=https://zenodo.org/record/8290055/files/$WEIGHTS_FN
RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}
RUN unzip ${WEIGHTS_DIR}${WEIGHTS_FN} -d ${WEIGHTS_DIR}
RUN rm ${WEIGHTS_DIR}${WEIGHTS_FN}

# specify nnunet specific environment variables
ENV WEIGHTS_FOLDER=$WEIGHTS_DIR
ENV CUDA_VISIBLE_DEVICES all

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6

# Copy scripts and config
COPY app/src/* /app/
COPY app/default.yml /app/

# Execute the script
ENTRYPOINT ["python3", "run.py", "--config", "default.yml"]
