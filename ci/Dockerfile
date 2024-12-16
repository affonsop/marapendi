FROM continuumio/miniconda3:latest
SHELL ["/bin/bash", "-c"]

# Installing c and c++ compilers
RUN apt-get update --fix-missing && apt-get install -y gcc make g++

# Creating Conda environment
ADD /ci/conda_env.yml /tmp/conda_env.yml
RUN conda env create -f /tmp/conda_env.yml

ADD . /buildingmodel
WORKDIR /buildingmodel

# Installing python package
RUN source activate buildingmodel && pip install .
RUN source activate buildingmodel && python -c "import coulomb"
