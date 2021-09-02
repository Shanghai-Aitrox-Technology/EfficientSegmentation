FROM nvcr.io/nvidia/pytorch:20.08-py3
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-setuptools \
        build-essential \
        && \
        apt-get clean && \
        python -m pip install --upgrade pip

WORKDIR /workspace
COPY ./   /workspace
RUN pip install pip -U
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# XXX: IN CASE pip install SimpleITK/scipy... and other pip package fail!
RUN pip install pip_packages/SimpleITK-1.1.0-cp36-cp36m-manylinux1_x86_64.whl
RUN pip install pip_packages/scikit_learn-0.24.2-cp36-cp36m-manylinux2010_x86_64.whl

RUN pip3 install --no-cache-dir lmdb \
                                pynvml \
                                prefetch-generator \
                                tensorboardX \
                                yacs \
                                connected-components-3d \
                                fastremap \
                                batchgenerators \
                                matplotlib \
                                scipy \
                                pydicom \
                                thop

# install Nvidia Apex
RUN tar -xvf apex.tar.gz
WORKDIR /workspace/apex
RUN pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
WORKDIR /workspace

#RUN pip install -e .
RUN rm -rf pip_packages
RUN rm -rf apex
CMD ["bash", "predict.sh"]
