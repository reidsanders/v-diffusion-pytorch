## conda install
#curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh

## TODO run installation of v-diffusion-pytorch
## move to outer dir

#git clone https://github.com/samoshkin/tmux-config.giit
git clone --branch patch-1 https://github.com/computertoucher/tmux-config.git
./tmux-config/install.sh

cp ./.inputrc ~/

printf 'export XRT_TPU_CONFIG="localservice;0;localhost:51011"\n' >> ~/.bashrc
#printf 'export PT_XLA_DEBUG=1' >> ~/.bashrc
printf 'export XLA_USE_BF16=1\n' >> ~/.bashrc

pip install -r ../requirements.txt
#pip3 install torch_xla[tpuvm] -f https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.10-cp38-cp38-linux_x86_64.whl ## TODO try setting in requrements.txt
