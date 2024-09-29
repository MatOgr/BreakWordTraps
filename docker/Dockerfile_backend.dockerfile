FROM python:3.12.6-bookworm

RUN apt-get update && apt-get install -qqy --no-install-recommends libopengl0 libopengl-dev libgl1-mesa-glx

COPY . /przemowa_backend
WORKDIR /przemowa_backend

RUN pip install -e .

EXPOSE 8888
ENTRYPOINT ["bwt", "run-backend", "127.0.0.1:8889", "127.0.0.1:8890", "--host", "0.0.0.0", "--port", "8888"]
