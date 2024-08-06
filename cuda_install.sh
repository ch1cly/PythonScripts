sudo apt-get remove --purge '^nvidia-.*'
sudo apt autoremove
dpkg -l | grep -i nvidia

sudo add-apt-repository ppa:graphics-drivers/ppa

sudo apt update
apt search nvidia-driver


sudo apt install nvidia-driver-550-server
sudo apt install nvidia-cuda-toolkit
sudo apt install nvidia-cuda-dev

sudo apt install nvidia-container-toolkit
