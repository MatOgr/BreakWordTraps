FROM python:3.12.6-bookworm

RUN apt-get update && apt-get install -qqy --no-install-recommends ffmpeg libopengl0 libopengl-dev libgl1-mesa-glx

COPY . /przemowa_asr
WORKDIR /przemowa_asr

RUN pip install -e '.[asr]'

EXPOSE 8889
ENTRYPOINT ["bwt", "run-subservice", "asr", "--host", "127.0.0.1", "--port", "8889", "--llm-server-address", "127.0.0.1:8891"]
