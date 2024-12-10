# Usa Miniconda como base
FROM continuumio/miniconda3:4.12.0

# Establece el directorio de trabajo
WORKDIR /app

# Instala herramientas necesarias para la compilación de Dlib
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    cmake \
    g++ \
    make \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copia el archivo environment.yml al contenedor
COPY environment.yml /app/environment.yml

# Crea el entorno Conda usando environment.yml
RUN conda env create -f /app/environment.yml && \
    conda clean --all --yes

# Configura el entorno por defecto
ENV PATH /opt/conda/envs/skillsevaluation/bin:$PATH
ENV CONDA_DEFAULT_ENV=skillsevaluation

# Activa el entorno por defecto al iniciar
SHELL ["conda", "run", "-n", "skillsevaluation", "/bin/bash", "-c"]

# Instala Dlib desde pip sin modificar las dependencias principales
RUN conda run -n skillsevaluation pip install dlib

# Copia el código fuente al contenedor
COPY src /app/src
