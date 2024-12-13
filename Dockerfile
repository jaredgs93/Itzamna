# Usa Miniconda como base
FROM continuumio/miniconda3:4.12.0

# Establece el directorio de trabajo
WORKDIR /app

# Instala herramientas necesarias para la compilación de Dlib
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    cmake \
    g++ \
    make \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copia los archivos de configuración para los entornos
COPY environment.yml /app/environment.yml
COPY environment-streamlit.yml /app/environment-streamlit.yml

# Crea el entorno Conda principal
RUN conda env create -f /app/environment.yml && \
    conda clean --all --yes

# Crea el entorno Conda para Streamlit
RUN conda env create -f /app/environment-streamlit.yml && \
    conda clean --all --yes

# Configura el entorno principal por defecto
ENV PATH /opt/conda/envs/skillsevaluation/bin:$PATH
ENV CONDA_DEFAULT_ENV=skillsevaluation

# Activa el entorno por defecto al iniciar
SHELL ["conda", "run", "-n", "skillsevaluation", "/bin/bash", "-c"]

# Instala Dlib en el entorno principal
RUN conda run -n skillsevaluation pip install dlib

# Copia el código fuente al contenedor
COPY src /app/src

# Configurar el PYTHONPATH para incluir `/app/src/app`
ENV PYTHONPATH=/app/src/app:$PYTHONPATH

# Cambiar el directorio de trabajo a `/app/src/app` para ejecutar la API
WORKDIR /app/src/app

# Ejecutar uvicorn con la configuración adecuada
CMD ["conda", "run", "-n", "skillsevaluation", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
