# pip install --upgrade pip
pip install -r requirements.txt

pip install gym==0.10.9
pip install gym[atari]

cd src
rm -rf atari-model-zoo/
git clone https://github.com/uber-research/atari-model-zoo.git
cd atari-model-zoo
python3 setup.py install
cd ../..

pwd=$(pwd)

pip uninstall lucid

wget -P $pwd https://github.com/tensorflow/lucid/archive/v0.3.9-alpha.tar.gz
mv v0.3.9-alpha.tar.gz lucid-0.3.9-alpha.tar.gz
tar -xvf lucid-0.3.9-alpha.tar.gz

cd lucid-0.3.9-alpha/
sed -i 's/0.3.8/0.3.9a/g' setup.py
python3 setup.py install

cd ../src