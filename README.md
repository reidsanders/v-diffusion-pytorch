# v-diffusion-pytorch

v objective diffusion inference code for PyTorch, by Katherine Crowson ([@RiversHaveWings](https://twitter.com/RiversHaveWings)) and Chainbreakers AI ([@jd_pressman](https://twitter.com/jd_pressman)).

The models are denoising diffusion probabilistic models (https://arxiv.org/abs/2006.11239), which are trained to reverse a gradual noising process, allowing the models to generate samples from the learned data distributions starting from random noise. The models are also trained on continuous timesteps. They use the 'v' objective from Progressive Distillation for Fast Sampling of Diffusion Models (https://openreview.net/forum?id=TIdIXIpzhoI). Guided diffusion sampling scripts (https://arxiv.org/abs/2105.05233) are included, specifically CLIP guided diffusion. This repo also includes a diffusion model conditioned on CLIP text embeddings that supports classifier-free guidance (https://openreview.net/pdf?id=qw8AKxfYbI), similar to GLIDE (https://arxiv.org/abs/2112.10741). Sampling methods include DDPM, DDIM (https://arxiv.org/abs/2010.02502), and PRK/PLMS (https://openreview.net/forum?id=PlKWVd2yBkY).

Thank you to [stability.ai](https://www.stability.ai) for compute to train these models!

## Dependencies

- PyTorch ([installation instructions](https://pytorch.org/get-started/locally/))

- requests, tqdm (install with `pip install`)

- CLIP (https://github.com/openai/CLIP), and its additional pip-installable dependencies: ftfy, regex. **If you `git clone --recursive` this repo, it should fetch CLIP automatically.**

## Model checkpoints:

- [CC12M_1 CFG 256x256](https://v-diffusion.s3.us-west-2.amazonaws.com/cc12m_1_cfg.pth), SHA-256 `4fc95ee1b3205a3f7422a07746383776e1dbc367eaf06a5b658ad351e77b7bda`

A 602M parameter CLIP conditioned model trained on [Conceptual 12M](https://github.com/google-research-datasets/conceptual-12m) for 3.1M steps and then fine-tuned for classifier-free guidance for 250K additional steps. **This is the recommended model to use.**

- [CC12M_1 256x256](https://v-diffusion.s3.us-west-2.amazonaws.com/cc12m_1.pth), SHA-256 `63946d1f6a1cb54b823df818c305d90a9c26611e594b5f208795864d5efe0d1f`

As above, before CFG fine-tuning. The model from the original release of this repo.

- [YFCC_1 512x512](https://v-diffusion.s3.us-west-2.amazonaws.com/yfcc_1.pth), SHA-256 `a1c0f6baaf89cb4c461f691c2505e451ff1f9524744ce15332b7987cc6e3f0c8`

A 481M parameter unconditional model trained on a 33 million image original resolution subset of [Yahoo Flickr Creative Commons 100 Million](http://projects.dfki.uni-kl.de/yfcc100m/).

- [YFCC_2 512x512](https://v-diffusion.s3.us-west-2.amazonaws.com/yfcc_2.pth), SHA-256 `69ad4e534feaaebfd4ccefbf03853d5834231ae1b5402b9d2c3e2b331de27907`

A 968M parameter unconditional model trained on a 33 million image original resolution subset of [Yahoo Flickr Creative Commons 100 Million](http://projects.dfki.uni-kl.de/yfcc100m/).

## Sampling

### Example

If the model checkpoint for cc12m_1_cfg is stored in `checkpoints/`, the following will generate four images:

```
./cfg_sample.py "the rise of consciousness":5 -n 4 -bs 4 --seed 0
```

If they are somewhere else, you need to specify the path to the checkpoint with `--checkpoint`.


### Colab

There is a cc12m_1_cfg Colab (a simplified version of `cfg_sample.py`) [here](https://colab.research.google.com/drive/1TBo4saFn1BCSfgXsmREFrUl3zSQFg6CC), which can be used for free.

### CFG sampling (best, but only cc12m_1_cfg supports it)

```
usage: cfg_sample.py [-h] [--images [IMAGE ...]] [--batch-size BATCH_SIZE]
                     [--checkpoint CHECKPOINT] [--device DEVICE] [--eta ETA] [--init INIT]
                     [--method {ddpm,ddim,prk,plms,pie,plms2}] [--model {cc12m_1_cfg}] [-n N]
                     [--seed SEED] [--size SIZE SIZE] [--starting-timestep STARTING_TIMESTEP]
                     [--steps STEPS]
                     [prompts ...]
```

`prompts`: the text prompts to use. Weights for text prompts can be specified by putting the weight after a colon, for example: `"the rise of consciousness:5"`. A weight of 1 will sample images that match the prompt roughly as well as images usually match prompts like that in the training set. The default weight is 3.

`--batch-size`: sample this many images at a time (default 1)

`--checkpoint`: manually specify the model checkpoint file

`--device`: the PyTorch device name to use (default autodetects)

`--eta`: set to 0 (the default) while using `--method ddim` for deterministic (DDIM) sampling, 1 for stochastic (DDPM) sampling, and in between to interpolate between the two.

`--images`: the image prompts to use (local files or HTTP(S) URLs). Weights for image prompts can be specified by putting the weight after a colon, for example: `"image_1.png:5"`. The default weight is 3.

`--init`: specify the init image (optional)

`--method`: specify the sampling method to use (DDPM, DDIM, PRK, PLMS, PIE, or PLMS2) (default PLMS). DDPM is the original SDE sampling method, DDIM integrates the probability flow ODE using a first order method, PLMS is fourth-order pseudo Adams-Bashforth, and PLMS2 is second-order pseudo Adams-Bashforth. PRK (fourth-order Pseudo Runge-Kutta) and PIE (second-order Pseudo Improved Euler) are used to bootstrap PLMS and PLMS2 but can be used on their own if you desire (slow).

`--model`: specify the model to use (default cc12m_1_cfg)

`-n`: sample until this many images are sampled (default 1)

`--seed`: specify the random seed (default 0)

`--starting-timestep`: specify the starting timestep if an init image is used (range 0-1, default 0.9)

`--size`: the output image size (default auto)

`--steps`: specify the number of diffusion timesteps (default is 50, can be lower for faster but lower quality sampling, must be much higher with DDIM and especially DDPM)


### CLIP guided sampling (all models)

```
usage: clip_sample.py [-h] [--images [IMAGE ...]] [--batch-size BATCH_SIZE]
                      [--checkpoint CHECKPOINT] [--clip-guidance-scale CLIP_GUIDANCE_SCALE]
                      [--cutn CUTN] [--cut-pow CUT_POW] [--device DEVICE] [--eta ETA]
                      [--init INIT] [--method {ddpm,ddim,prk,plms,pie,plms2}]
                      [--model {cc12m_1,cc12m_1_cfg,yfcc_1,yfcc_2}] [-n N] [--seed SEED]
                      [--size SIZE SIZE] [--starting-timestep STARTING_TIMESTEP] [--steps STEPS]
                      [prompts ...]
```

`prompts`: the text prompts to use. Relative weights for text prompts can be specified by putting the weight after a colon, for example: `"the rise of consciousness:0.5"`.

`--batch-size`: sample this many images at a time (default 1)

`--checkpoint`: manually specify the model checkpoint file

`--clip-guidance-scale`: how strongly the result should match the text prompt (default 500). If set to 0, the cc12m_1 model will still be CLIP conditioned and sampling will go faster and use less memory.

`--cutn`: the number of random crops to compute CLIP embeddings for (default 16)

`--cut-pow`: the random crop size power (default 1)

`--device`: the PyTorch device name to use (default autodetects)

`--eta`: set to 0 (the default) while using `--method ddim` for deterministic (DDIM) sampling, 1 for stochastic (DDPM) sampling, and in between to interpolate between the two.

`--images`: the image prompts to use (local files or HTTP(S) URLs). Relative weights for image prompts can be specified by putting the weight after a colon, for example: `"image_1.png:0.5"`.

`--init`: specify the init image (optional)

`--method`: specify the sampling method to use (DDPM, DDIM, PRK, PLMS, PIE, or PLMS2) (default DDPM). DDPM is the original SDE sampling method, DDIM integrates the probability flow ODE using a first order method, PLMS is fourth-order pseudo Adams-Bashforth, and PLMS2 is second-order pseudo Adams-Bashforth. PRK (fourth-order Pseudo Runge-Kutta) and PIE (second-order Pseudo Improved Euler) are used to bootstrap PLMS and PLMS2 but can be used on their own if you desire (slow).

`--model`: specify the model to use (default cc12m_1)

`-n`: sample until this many images are sampled (default 1)

`--seed`: specify the random seed (default 0)

`--starting-timestep`: specify the starting timestep if an init image is used (range 0-1, default 0.9)

`--size`: the output image size (default auto)

`--steps`: specify the number of diffusion timesteps (default is 1000, can lower for faster but lower quality sampling)

## Training (on TPU)

The training code is currently focused on training on TPUs (thanks Tensorflow Research Cloud!).

Currently it is for training cc12m only. 

You can follow tpu creation and installation as in https://github.com/reidsanders/cloud-setup-scripts. conda is recommended, but you can use whatever you prefer.

If you are not running those scripts you will still want to set the following environment variables:

```sh
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export XLA_USE_BF16=1
```

And install the appropriate torch_xla for your torch:
```sh
pip3 install torch_xla[tpuvm] -f https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.10-cp38-cp38-linux_x86_64.whl
```

Then you can start a training run:

```sh
python3 cc12m_1_cfg_train.py --train_set ~/datasets/goodbot/ --demo_prompts demo-prompts.txt --batchsize 3 --dataset_mode text --project_name goodbot-diffusion --max_epochs 100 --checkpoint checkpoints/cc12m_1_cfg.pth --lr 1e-5 --val_check_interval .5 --scheduler exponentiallr --gamma .99 --accumulate_grad_batches 8
```

You'll need to paste your wandb key the first time.

### Config

```sh
$ python3 cc12m_1_cfg_train.py -h
2022-02-27 04:32:51.146314: E tensorflow/core/framework/op_kernel.cc:1676] OpKernel ('op: "TPURoundRobin" device_type: "CPU"') for unknown op: TPURoundRobin
2022-02-27 04:32:51.146380: E tensorflow/core/framework/op_kernel.cc:1676] OpKernel ('op: "TpuHandleToProtoKey" device_type: "CPU"') for unknown op: TpuHandleToProtoKey
usage: cc12m_1_cfg_train.py [-h] --train_set TRAIN_SET [--val_set VAL_SET] [--test_set TEST_SET] --demo_prompts DEMO_PROMPTS [--checkpoint CHECKPOINT] [--batchsize BATCHSIZE]
                            [--scheduler_epochs SCHEDULER_EPOCHS] [--imgsize IMGSIZE] [--dataset_mode [{conceptual,drawtext,text,danbooru,goodbot}]] [--project_name PROJECT_NAME] [--lr LR]
                            [--gamma GAMMA] [--scheduler [{cosineannealingwarmrestarts,exponentiallr,onecyclelr}]] [--restore_train_state]

optional arguments:
  -h, --help            show this help message and exit
  --train_set TRAIN_SET
                        the training set location
  --val_set VAL_SET     the val set location
  --test_set TEST_SET   the test set location
  --demo_prompts DEMO_PROMPTS
                        the demo prompts
  --checkpoint CHECKPOINT
                        load checkpoint file path
  --batchsize BATCHSIZE
                        batchsize for training
  --scheduler_epochs SCHEDULER_EPOCHS
                        epochs to pass to lr scheduler
  --imgsize IMGSIZE     Image size in pixels. Assumes square image
  --dataset_mode [{conceptual,drawtext,text,danbooru,goodbot}]
                        choose dataset loader mode (default: drawtext)
  --project_name PROJECT_NAME
                        project name for logging
  --lr LR               starting lr
  --gamma GAMMA         exponential decay gamma for lr
  --scheduler [{cosineannealingwarmrestarts,exponentiallr,onecyclelr}]
                        choose dataset loader mode (default: None)
  --restore_train_state
                        restore lightning training state
```
You can ignore the `TPURoundRobin` and `TpuHandleToProtoKey` warnings. 

If you do not pass a val_set it will split the train_set into a train and val set for you. 

In addition to the explicit args you can pass all the pytorch lightning [Trainer parameters](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#methods). For example `accumulate_grad_batches` or `fast_dev_run`

### Dataloaders

This includes a few dataloaders for various experiments. They are not substantially different, generally just loading labels from json, but may be useful.

### Caveats

- This is fairly hacky and doubtless has bugs. Currently trying to load a checkpoint while using CosineAnnealingWarmRestarts crashes since the 'initial_lr' is not set.

- Note that wandb does not play well with multiple cores, hence the use of wandb service experiment. It is still rather broken, and cannot log output or commands, so this includes a basic command logger. `wandb.init()` and `wandb.config.update()` do not work. If you want to add something to the config you need to set it manually with

```python
wandb.config.yourvalue = 2345
```

- Lightning tuner and profiler does not work.

You can also download some pretrained models for some experiments here. See also [dataset creation scripts]https://github.com/reidsanders/dataset-creation-scripts for text and emoji dataset generation, and [danbooru utility]https://github.com/reidsanders/danbooru-utility for face recognition and filtering on gwern's danbooru dataset.

This is formatted with black.

### Acknowledgements

Thanks to @kcrawson for the original code. Also thanks to Tensorflow Research Cloud for the tpu credits.