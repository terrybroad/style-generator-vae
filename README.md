# Variational autoencoder with style-based generator

PyTorch implementation of a variational autoencoder that utilises a style-based generator [Karras et al. 18](https://arxiv.org/abs/1812.04948). There are two training files,
a basic (beta)vae training regime, and an implementation of the aggressive inference training regime [He et al. 19](https://openreview.net/pdf?id=rylDfnCqF7).
Second implementation was done as basic VAE training regime suffers from posterior collapse. 

Implementation of style-based generator is based on: https://github.com/rosinality/style-based-gan-pytorch

Work in progress.

To train:
```python3 train_vae.py /path/to/data``` or ```python3 train_aggressive_inf_vae.py /path/to/data```
