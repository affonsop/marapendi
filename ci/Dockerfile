FROM continuumio/miniconda3:latest
SHELL ["/bin/bash", "-c"]

# Installing c and c++ compilers
RUN apt-get update --fix-missing && apt-get install -y gcc make g++

# Creating Conda environment
ADD /ci/conda_env.yml /tmp/conda_env.yml
RUN conda env create -f /tmp/conda_env.yml

ADD . /marapendi
WORKDIR /marapendi

# Make the marapendi conda env the default for RUN/CMD/ENTRYPOINT, so
# `docker run marapendi:ci <cmd>` works without an explicit `source activate`.
ENV PATH=/opt/conda/envs/marapendi/bin:$PATH

# Installing python package
RUN pip install .
RUN python -c "import marapendi"
