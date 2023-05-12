FROM python:3.9
FROM nvidia/cuda:11.6.2-base-ubuntu20.04

WORKDIR /src/columbia_skip_doc

RUN apt update && \
    apt install --no-install-recommends -y build-essential python3 python3-pip && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

COPY . ./

ENTRYPOINT ["streamlit", "run", "streamlit_app.py","--server.port=8501", "--server.address=0.0.0.0"]


