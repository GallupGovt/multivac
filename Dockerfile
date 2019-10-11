# use ubuntu as the base image; install R and Python on top
FROM ubuntu:latest

# avoid humna input for geography and stuff
ENV DEBIAN_FRONTEND=noninteractive

# install R and python
RUN apt-get update && apt-get install -y --no-install-recommends build-essential r-base r-cran-randomforest python3.7 python3-pip python3-setuptools python3-dev git

# set working directory to /app
WORKDIR /app

# copy requirements over to application
COPY requirements.txt /app/requirements.txt

# dummy R installation for testing purposes
RUN Rscript -e "install.packages('data.table')"

# install Python dependencies
RUN pip3 install -r requirements.txt

# install do
RUN git clone https://github.com/thunlp/OpenKE && \
    cd OpenKE && \
    sh make.sh

COPY . /app

ENTRYPOINT ["python3"]

# run
CMD ["-m", "http.server", "5050"]
