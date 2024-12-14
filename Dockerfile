FROM continuumio/miniconda3:4.12.0

# Sets the working directory
WORKDIR /app

# Install tools needed for Dlib compilation
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

# Copy the configuration files for the environments
COPY environment.yml /app/environment.yml
COPY environment-streamlit.yml /app/environment-streamlit.yml

# Create the main Conda environment
RUN conda env create -f /app/environment.yml && \
    conda clean --all --yes

# Create the Conda environment for Streamlit
RUN conda env create -f /app/environment-streamlit.yml && \
    conda clean --all --yes

# Set the default main environment
ENV PATH /opt/conda/envs/skillsevaluation/bin:$PATH
ENV CONDA_DEFAULT_ENV=skillsevaluation

# Activates the default environment at start-up
SHELL ["conda", "run", "-n", "skillsevaluation", "/bin/bash", "-c"]

# Install Dlib in the main environment
RUN conda run -n skillsevaluation pip install dlib

# Copy the source code to the container
COPY src /app/src

# Configure the PYTHONPATH to include `/app/src/app`.
ENV PYTHONPATH=/app/src/app:$PYTHONPATH

# Change working directory to `/app/src/app` to run the API
WORKDIR /app/src/app

# Running uvicorn
CMD ["conda", "run", "-n", "skillsevaluation", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
