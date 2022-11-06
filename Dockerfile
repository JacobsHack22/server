
# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.8-slim-buster 

# RUN apt-get update -y
# RUN apt-get install -y python-pip
 
COPY . /app
 
# Create and change to the app directory.
WORKDIR /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
 
RUN pip install --no-cache-dir -r requirements.txt
 
RUN chmod 444 main.py
RUN chmod 444 requirements.txt


 
# Service must listen to $PORT environment variable.
# This default value facilitates local development.
ENV PORT 8080
 
# Run the web service on container startup.
CMD [ "python", "main.py" ]