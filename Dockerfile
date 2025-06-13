# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8888

CMD ["bash"]