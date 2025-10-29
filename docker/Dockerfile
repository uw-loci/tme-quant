# Dockerfile for tme-quant
# This creates a containerized environment for tme-quant
FROM continuumio/miniconda3:latest

LABEL maintainer="Eliceiri Lab"
LABEL description="tme-quant: Python translation of CurveAlign for collagen fiber analysis"

# Set working directory
WORKDIR /app

# Install system dependencies for building FFTW and CurveLab
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install FFTW 2.1.5
RUN cd /tmp && \
    wget http://www.fftw.org/fftw-2.1.5.tar.gz && \
    tar xzf fftw-2.1.5.tar.gz && \
    cd fftw-2.1.5 && \
    ./configure --prefix=/usr/local/fftw-2.1.5 --disable-fortran && \
    make -j$(nproc) && \
    make install && \
    ln -sf include fftw && \
    cd /tmp && \
    rm -rf fftw-2.1.5*

# Environment variables
ENV FFTW=/usr/local/fftw-2.1.5
ENV CPPFLAGS="-I${FFTW}/include"
ENV LDFLAGS="-L${FFTW}/lib"

# Copy project files (context is parent directory)
COPY . /app

# Note: CurveLab must be provided by mounting it at runtime
# Example: docker run -v /path/to/CurveLab-2.1.3:/app/curvelab ...
# The user must download CurveLab separately due to licensing

# Create conda environment
RUN conda create -y -n tme-quant python=3.11 && \
    echo "conda activate tme-quant" >> ~/.bashrc

# Set default command
CMD ["/bin/bash"]

