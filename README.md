# HF LM training

## What this repo provides

This is a wrapper over the standard HF language modeling training code, with the following features:
* Catch the SLURM job ending signal and save the model checkpoint before the job timeouts.
* Customized llama implementation (from Alex Wettig) with variable length flash attention.
* Implementation of hybrid architectures, where the efficient attention replacement = sliding window attention, SRU, Mamba.
* Support of Mosaic's streaming dataset.
* Support of sequence parallelism.

## Environment

```bash
pip install -r requirements.txt
```

or an easier way is to install all the lastest versions of the following packages (may not be complete; if missing any, just install the latest version):
```bash
pip install torch transformers datasets accelerate mosaicml-streaming wandb tqdm evaluate simple_parsing
pip install flash-attn --no-build-isolation
```

If you either need to install new packages, or activating Tianyu's environment does not work for you, e.g. due to broken paths, then you can create your own Mamba environment. First install Mamba, following [directions here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

Then create a new Mamba environemnt, install the required packages as outlined in the above command.

As SRU requires specific versions of gcc and C++, you may need to update your gcc and C++ version, you can do so from within the Mamba environement, with: `mamba install -c conda-forge gcc=10` and `mamba install -c conda-forge gxx=10`.



If you don't need to add any new packages, you can use Tianyu's environment:
```bash
source /scratch/gpfs/tianyug/lm-training/.venv-latest-transformers-torch/bin/activate
```

### The SRU environment

SRU requires just-in-time (JIT) compilation, which can be tricky to set up (requires CUDA compilation environment, correct GCC version, etc.). The easiest way is to use mamba (a light veresion of conda) to install and setup the environment. Check the SRU repo. 

If you don't need to add any new packages, you can use Tianyu's environment:
```bash
source /scratch/gpfs/tianyug/miniforge3/etc/profile.d/mamba.sh
mamba activate sru
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/gpfs/tianyug/miniforge3/lib
export SRU=1 # This is to tell our code to load the SRU module at the beginning; otherwise there is some weird environment problem
```

## How to run it on a SLURM cluster (Della PLI as an example)

### Use interactive job (for debugging)

First get 1 or 8 GPUs with your interactive job

```bash
salloc -N 1 --ntasks-per-node 8 --gres=gpu:8 --cpus-per-task 10 --mem 800G  --time 2:00:00 -p pli-c
```

For a single GPU debug, run
```bash
export WANDB_MODE=disabled
python run_clm.py --run_config {path to the config file}
```

For a multi-GPU debug, copy `torchrun_launcher_debug.sh` to `torchrun_launcher.sh`, add proper environment, and run
```bash
NUM_GPU=8 bash torchrun_launcher.sh --run_config {path to the config file}
```

### Use sbatch job (multi-GPU or multi-node)

First, copy `srun_launcher.sh.example` to `srun_launcher.sh`, modify it to setup the environment, and run
```bash
# eclude: exclude those nodes. Currently those nodes on PLI have weird NCCL problems.
# nodes: how mnay nodes to use
# jobname: SLURM jobname. This script allows "dependency" jobs, i.e., you can submit multiple jobs with the same name and they will run sequentially.
# repeat: how many times to repeat the job
# runtime: how many hours. 
exclude="" nodes=4 jobname=dclm-baseline repeat=1 runtime=24 bash run_slurm_train_dep_pli.sh --run_config run_configs/example-1.6b-dclm.yaml
```

You can read the scripts and play with them for customization.


## Configs

You can find example configs under `run_configs`. You can specify the config by `--run_config {path to the config file}`. You can add more arguments after this to override the arguments in the config file. See `run_configs/example-1.6b-dclm.yaml` and `run_clm.py` for possible arguments.

