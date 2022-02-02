## conda install
#curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh


#git clone https://github.com/samoshkin/tmux-config.giit
git clone --branch patch-1 https://github.com/computertoucher/tmux-config.git
./tmux-config/install.sh

cp ./.inputrc ~/

printf 'export XRT_TPU_CONFIG="localservice;0;localhost:51011"' >> ~/.bashrc
printf 'export PT_XLA_DEBUG=1' >> ~/.bashrc
