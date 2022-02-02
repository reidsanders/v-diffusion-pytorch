curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py > /dev/null
python3 pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev > /dev/null
pip3 install pytorch-lightning > /dev/null

