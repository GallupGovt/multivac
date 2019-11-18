# MULTIVAC Installation Guide

### Installation Requirements
MULTIVAC can be most easily and cleanly installed using `docker`. To use this method, Docker Desktop is required for launching the system on your local machine. Docker Desktop can be set up easily for either Mac or Windows machines with resources found at the following links:
* For Mac users: https://docs.docker.com/docker-for-mac/install/
* For Windows users: https://docs.docker.com/docker-for-windows/install/

MULTIVAC makes use of multiple linked docker containers, so along with Docker Desktop, users will need to have set up `docker-compose`. Mac, Windows, and Linux instructions for installation can be found here: 
* Docker Compose: https://docs.docker.com/compose/install/

### Downloading and Deploying MULTIVAC
The first step is to clone this MULTIVAC repository from GitHub. With Git also locally <a href="https://git-scm.com/book/en/v2/Getting-Started-Installing-Git">installed</a>:
* Run the following command in your preferred directory: `git clone https://github.com/GallupGovt/multivac.git`
* Next, change into the MULTIVAC directory you just created and run: `docker-compose up`

This command will download and build the resources MULTIVAC depends on: Stanford CoreNLP, Grobid Publication Parsing, and Jupyter Notebook Viewer, as well as the core MULTIVAC system itself. This process will take some time on first use, and require well over 10 GB of hard drive space, so please plan accordingly. 

### Basic Operations
In order to see the running processes under Docker, you can use the `docker ps` command. You should see a running container named *multivac_multivac:latest*. This is the root source of our project. To interact with our code and system, you may use `docker exec -it {container-of-multivac-id} {command}`(i.e. `docker exec -it abd35789sbd2 python3 querygan_pyt.py --cuda`). You can also access our web application through port 5000 of your machine, i.e. http://0.0.0.0:5000 or http://your.ip.add:5000 if on a VM. 

To run any docker commands in the background, add the flag `-d` to your command. Once the system is built, you can always start and stop it with the commands `docker-compose start` and `docker-compose stop`. 
