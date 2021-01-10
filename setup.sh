sudo add-apt-repository ppa:deadsnakes/ppa
sudo add-apt-repository universe
sudo apt -q update
sudo apt -q install software-properties-common
sudo apt -q install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
sudo apt -q install python3.8
sudo apt -q install python3-pip
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10
pip3 install virtualenv
virtualenv -p /usr/bin/python3.8 venv
source ./venv/bin/activate
pip install -r requirements.txt
python -m pip install --upgrade pip
nohup uvicorn app:app --reload > service.log