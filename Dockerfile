FROM python:3.11.1-slim

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
