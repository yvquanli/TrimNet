FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*
    

COPY . .
COPY requirements.txt .
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
RUN pip3 install -r requirements.txt


WORKDIR .

CMD ["python3", "trimnet_drug/source/run.py"]