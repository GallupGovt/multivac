# use ubuntu as the base image; install R and Python on top
FROM ubuntu:latest

# avoid human input for geography and stuff
ENV DEBIAN_FRONTEND=noninteractive

# # Confirm you have no swap
# RUN sudo swapon -s

# # Allocate 1GB (or more if you wish) in /swapfile
# sudo fallocate -l 1G /swapfile

# # Make it secure
# sudo chmod 600 /swapfile
# ls -lh /swapfile

# # Activate it
# sudo mkswap /swapfile
# sudo swapon /swapfile

# # Confirm again there's indeed more memory now
# free -m
# sudo swapon -s

# # Configure fstab to use swap when instance restart
# sudo nano /etc/fstab

# # Add this line to /etc/fstab, save and exit
# /swapfile   none    swap    sw    0   0

# # Change swappiness to 10, so that swap is used only when 10% RAM is unused
# # The default is too high at 60
# echo 10 | sudo tee /proc/sys/vm/swappiness
# echo vm.swappiness = 10 | sudo tee -a /etc/sysctl.conf

# install R and python
RUN apt-get update && apt-get install -y --no-install-recommends build-essential r-base python3.7 python3-pip python3-setuptools python3-dev git

# copy requirements over to application
COPY requirements.txt /multivac/requirements.txt

WORKDIR /multivac

# set up bdist_wheel
RUN pip3 install wheel
RUN pip3 install setuptools

# env setup for torch + other requirements
RUN pip3 install torch==1.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r requirements.txt


RUN git clone https://github.com/thunlp/OpenKE && cd OpenKE/openke && sh make.sh

COPY . /multivac

ENV PYTHONPATH "${PYTHONPATH}:/multivac"

EXPOSE 5000

CMD python3 app.py


### Look into this if issues with OpenKE sh (production image)
# https://forums.docker.com/t/best-practices-for-git-clone-make-etc-via-dockerfile-run/79152/3
