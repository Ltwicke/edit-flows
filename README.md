# Language Modeling with Edit Flows

A well-documented and typed PyTorch implementation of Edit Flows by Marten Lienen.

Edit Flows is a generative model for variable-length sequences described in [Edit Flows: Flow Matching with Edit Operations](https://arxiv.org/abs/2506.09018) by Marton Havasi, Brian Karrer, Itai Gat and Ricky Chen. I intend this repository to serve as the basis for further research with a working, albeit minimal, language modeling pipeline. The code in [editflows.py](./editflows/editflows.py) is a self-contained implementation of the model that can be dropped straight into other projects. Integrating it into a training framework requires just a small wrapper, for example [pretraining.py](./editflows/pretraining.py) for PyTorch Lightning.

This repository has served as the basis for the following papers and projects:
1. [Edit-Based Flow Matching for Temporal Point Processes](https://arxiv.org/abs/2510.06050) by David Lüdke\*, Marten Lienen\*, Marcel Kollovieh, Stephan Günnemann

If you build on this code, please let me know and I will add your project above.

<p align="center"><img src="figures/figure-1-havasi-et-al.png"><br />The Edit Flows sampling process (Figure 1, <a href="https://arxiv.org/abs/2506.09018">Havasi et al.</a>)</p>

## Results

I have trained a Llama 2 model on [text8](https://mattmahoney.net/dc/textdata.html) to verify the implementation. The model has about 180M parameters and trained on 4 A100s for 100k steps in about 10 hours.
```
$ ./sample.py -n 2 --seed 55 ./text8.ckpt
records supposing that us and u s were too avoided firing on the occasion the issue marked how much the french worked more on the issue some feel that they still is the very occasion they saw the united states s ships of which could become a carrier on their own islands with the british but the military itself announced the eight zero zero zero month army road national fleet went because france intended that roads would be all the time to break

oc the protection itself in the two ways all economic or religious pure can imply a political perspective notably in the language debekara p four three zero like is the sense of like it is such and does not appear to have the extent of effective or uncertain all its response must be verbrella the parma hogana dutch middleme journers arher or ed wood see also fugure fathers external
```
Of course, this is far from actual English, but it is sufficiently close for me to call it working. If you train the model for longer or on larger data, let me know!

To replicate the sample above, either download the checkpoint with
```
curl -L -o text8.ckpt https://huggingface.co/martenlienen/edit-flows/resolve/main/text8.ckpt
```
or create one yourself with
```
./train.py experiment=text8-180M
```

## Installation

We use [pixi](https://pixi.sh/) to easily set up reproducible environments. Install it with `curl -fsSL https://pixi.sh/install.sh | bash`, then install and activate the environment with

```sh
pixi shell
```

This gives you a GPU environment by default. For a machine without a GPU, activate the CPU environment instead with `pixi shell -e cpudev`.

## Training

Training is configured with [hydra](https://hydra.cc), so you can override any setting from the command line and browse the defaults in the [`config`](./config) directory. The datamodule downloads the text8 dataset and prepares the tokenizer automatically on the first run, so you can just start a training.

Any setting can be overridden inline, for example the batch size or the number of steps

```sh
./train.py experiment=text8-180M data.batch_size=256 trainer.max_steps=50000
```

Training logs to [Weights & Biases](https://wandb.ai) by default. To log to a local CSV file instead, add `logging=csv`.

Self-contained checkpoints are written to `checkpoints/<run-name>/<run-id>/`: in addition to the weights, each one also stores the model configuration and the SentencePiece tokenizer.

## Sampling

You can sample sequences from a trained model with

```sh
./sample.py -n <samples> path/to/file.ckpt
```

For more options, check [sample.py](./sample.py) or run `./sample.py --help`.
