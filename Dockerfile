# Usa Miniconda como base
FROM continuumio/miniconda3:4.12.0

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo environment.yml al contenedor
COPY environment.yml /app/environment.yml

# Crea un entorno Conda vacío con Python 3.9
RUN conda create --name skillsevaluation python=3.9 && \
    conda clean --all --yes

# Configura el entorno por defecto
ENV PATH /opt/conda/envs/skillsevaluation/bin:$PATH

# Copia el código fuente al contenedor
COPY src /app/src
