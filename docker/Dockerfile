# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG DLFW_VERSION
FROM nvcr.io/nvidia/pytorch:${DLFW_VERSION}-py3

# update repo info
RUN apt update -y && apt install -y libibmad5

# upgrade cmake
RUN apt remove cmake -y && \
    pip install cmake --upgrade

# install mpi4py
RUN SETUPTOOLS_USE_DISTUTILS=local pip install mpi4py

# install onnx
RUN pip install onnx onnxruntime onnxruntime-gpu

# hdf5 and h5py
ENV HDF5_VERSION=1.14.6
RUN cd /tmp && wget https://support.hdfgroup.org/releases/hdf5/v1_14/v1_14_6/downloads/hdf5-${HDF5_VERSION}.tar.gz && \
    gzip -cd hdf5-${HDF5_VERSION}.tar.gz | tar xvf - && \
    mkdir hdf5-${HDF5_VERSION}/build && cd hdf5-${HDF5_VERSION}/build && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/hdf5 \
    -DHDF5_ENABLE_OPTIMIZATION=ON \
    -DHDF5_ENABLE_DIRECT_VFD=ON \
    -DHDF5_ENABLE_ROS3_VFD=ON \
    -DHDF5_ENABLE_PARALLEL=ON \
    -DHDF5_TEST_API=ON \
    -DHDF5_TEST_VFD=ON \
    -DHDF5_TEST_PARALLEL=ON \
    .. && \
    make -j 8 && make install
RUN CC="mpicc" HDF5_MPI=ON H5PY_ROS3=1 H5PY_DIRECT_VFD=1 HDF5_DIR=/opt/hdf5 pip install --no-build-isolation --no-binary=h5py h5py==3.13.0
ENV PATH=/opt/hdf5/bin:${PATH}

# install cdsapi for downloading the dataset
RUN pip install cdsapi>=0.7.2

# install zarr and data stuff
RUN pip install more_itertools zarr xarray pandas gcsfs boto3

# moviepy imageio for wandb video logging
RUN pip install moviepy imageio

# other python stuff
RUN pip install --upgrade wandb ruamel.yaml tqdm progressbar2 jsbeautifier

# numba
RUN pip install numba
ENV NUMBA_DISABLE_CUDA=1

# scoring tools
RUN pip install xskillscore properscoring

# benchy
RUN pip install git+https://github.com/romerojosh/benchy.git

# some useful scripts from mlperf
RUN pip install --ignore-installed "git+https://github.com/NVIDIA/mlperf-common.git"

# torch-harmonics
ENV FORCE_CUDA_EXTENSION 1
ENV TORCH_CUDA_ARCH_LIST "8.0 8.6 9.0 10.0 12.0+PTX"
ENV HARMONICS_VERSION 0.8.0
RUN cd /opt && git clone https://github.com/NVIDIA/torch-harmonics.git && \
    cd torch-harmonics && \
    pip install --no-build-isolation -e .

# physicsnemo
RUN pip install git+https://github.com/NVIDIA/physicsnemo.git@v1.1.1

# copy source code
RUN mkdir -p /opt/makani
COPY config /opt/makani/config
COPY docker /opt/makani/docker
COPY data_process /opt/makani/data_process
COPY datasets /opt/makani/datasets
COPY makani /opt/makani/makani
COPY tests /opt/makani/tests
COPY pyproject.toml /opt/makani/pyproject.toml
COPY README.md /opt/makani/README.md
RUN cd /opt/makani && pip install -e .