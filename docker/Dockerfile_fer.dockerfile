FROM python:3.12.6-bookworm

RUN apt-get update && apt-get install -qqy --no-install-recommends libopengl0 libopengl-dev libgl1-mesa-glx

COPY . /przemowa_asr
WORKDIR /przemowa_asr

RUN pip install -e '.[fer]'

EXPOSE 8890
ENTRYPOINT ["bwt", "run-subservice", "fer", "--host", "127.0.0.1", "--port", "8890"]
