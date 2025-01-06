# Metadata Conditioning Accelerates Language Model Pre-training (MeCo)

[[Paper](https://arxiv.org/pdf/2501.01956)] [[HF Page](https://huggingface.co/collections/PrincetonPLI/meco-677bbbc3d5fbd8da65c6ee5a)]

This is the homepage for paper **Metadata Conditioning Accelerates Language Model Pre-training**. 

We propose a new pre-training method named **metadata conditioning then cooldown (MeCo)**: it conditions pre-training texts with their metadata (such as source URLs) by prepending the metadata to the corresponding documents; at the end of training, it switches to a cooldown phase with only the standard texts to enable inference without metadata.

MeCo significantly accelerates pre-training across different model scales (600M to 8B parameters) and training sources (C4, RefinedWeb, and DCLM). For instance, a 1.6B language model trained with MeCo matches the downstream task performance of standard pre-training while using 33% less data. 


![alt text](meco.png)


Authors: [Tianyu Gao](https://gaotianyu.xyz/about) (`tianyug@princeton.edu`), [Alexander Wettig](https://www.cs.princeton.edu/~awettig/), [Luxi He](https://lumos23.github.io/), [Yihe Dong](https://yihedong.me/), [Sadhika Malladi](https://www.cs.princeton.edu/~smalladi/), [Danqi Chen](https://www.cs.princeton.edu/~danqic/) 


## Quick Links

  - [Requirements](#requirements)
  - [Data](#data)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Downloading models](#downloading-models)
  - [Citation](#citation)


## Release Progress

- [x] Training code
- [x] Checkpoints
- [ ] Data and data preparation
- [ ] Training readme
- [ ] Evaluation readme

## Requirements

Coming soon!

## Data

Coming soon!

## Training
Coming soon!

## Evaluation

Coming soon!

## Downloading models

You can download the checkpoints in our experiments from our [Hugging Face collection](https://huggingface.co/collections/PrincetonPLI/meco-677bbbc3d5fbd8da65c6ee5a).

## Citation

```bibtex
@article{gao2025meco,
  title={Metadata Conditioning Accelerates Language Model Pre-training},
  author={Tianyu Gao and Alexander Wettig and Luxi He and Yihe Dong and Sadhika Malladi and Danqi Chen},
  journal={arXiv preprint arXiv:2501.01956},
  year={2025}
}
```
