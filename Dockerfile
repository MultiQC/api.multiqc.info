FROM python:3.12.1-slim

RUN apt-get update && apt-get install -y git

WORKDIR /code

# Don't bother creating .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# Don't buffer stdout and stderr
ENV PYTHONUNBUFFERED 1

# Install deps
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Run the app
COPY ./app /code/app
