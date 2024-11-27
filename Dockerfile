#FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9
#RUN apt-get -y update
#RUN apt-get install -y ffmpeg
#COPY ./requirements.txt /app/requirements.txt
#COPY ./app/datasets /app/datasets
#COPY ./app/myprosody /app/myprosody
#COPY ./app/Mysp /app/Mysp
#COPY ./app/temp /app/temp
#RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
#COPY ./app /app/app

FROM python:3.9
RUN apt -y update
RUN apt install -y ffmpeg
RUN apt install -y cmake
COPY ./requirements.txt /app/requirements.txt
COPY ./app /app/app
RUN pip install dlib
RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt
WORKDIR /app/app
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0","--port", "80", "--workers", "2"]
#CMD uvicorn main:app --reload --host 0.0.0.0 --port $PORT