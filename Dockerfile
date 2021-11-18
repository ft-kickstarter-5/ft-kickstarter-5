#Create a ubuntu base image with python 3 installed.
FROM python:3.10 as base

FROM base AS python-deps

#Set the working directory
WORKDIR /


# Install python dependencies in /.venv
COPY Pipfile .
COPY Pipfile.lock .

#Install the dependencies
RUN apt-get -y update
RUN apt-get update && apt-get install -y python3 python3-pip


# Install pipenv and compilation dependencies
RUN pip3 install pipenv
RUN apt-get update && apt-get install -y --no-install-recommends gcc
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy


# Install gunicorn requirements.txt
COPY requirements.txt .

# install requirements.txt
RUN pip3 install -r requirements.txt

FROM base AS runtime

# Copy virtual env from python-deps stage
COPY --from=python-deps /.venv /.venv
ENV PATH="/.venv/bin:$PATH"

# Create and switch to a new user
RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser

# Install application into container
COPY . .

#Expose the required port
EXPOSE 5000

#Run the command
CMD gunicorn kickstarter:app