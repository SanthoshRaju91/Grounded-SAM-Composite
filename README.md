# Grounded-SAM-Composite

1. Set CUDA HOME

example: `%env CUDA_HOME=/usr/local/cuda-12.6`

2. Install GroundingDINO

```shell
cd GroundingDINO
pip install -e .
```

3. Download weights for the GroundingDINO

```shell
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O grouding_dino_weights.pth
```

4. Install segment-anything

```shell
cd segment-anything
pip install -e .
```

5. Download SAM model checkpoint weights

```shell
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
