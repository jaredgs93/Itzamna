FROM tensorflow/tensorflow:2.10.0 

RUN apt update && apt install -y ffmpeg cmake git 

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./app /app/app
WORKDIR /app/app
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "80", "--workers", "2"]
