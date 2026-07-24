# AGNN Triple Attention for Few-Shot Learning

This repository contains the current working implementation of AGNN with triple attention for few-shot learning, plus a Streamlit demo for interactive inference on custom support images.

## What is included

- Training and evaluation entrypoint: `main_gnn.py`
- Backbone pretraining for custom datasets: `pretrain.py`
- Hybrid inference with support images and/or base prototypes: `inference.py`
- Streamlit demo app: `app.py`
- Demo launcher: `run_app.bat`

## Environment

Use a Python environment with the following packages installed:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `Pillow`
- `streamlit`
- `tensorboardX`
- `torchnet`

The repo also contains Windows-specific safeguards for OpenMP and NumPy/PyTorch checkpoint compatibility in `app.py`.

## Data Preparation

The repo is currently set up around custom few-shot workflows rather than the original paper dataset package.

For the Streamlit demo, the support set is read from `app/demo_support/<class_name>/` with images inside each class folder.

For `pretrain.py`, the dataset root should look like a standard image classification dataset:

```bash
dataset_root/
├── class_a/
│   ├── img1.jpg
│   └── img2.jpg
├── class_b/
│   └── ...
└── ...
```

`pretrain.py` also requires a `split.json` that defines the `train` and `val` class lists.

For `main_gnn.py`, use a dataset root and a config file from `config/` that matches your experiment.

## Training and Evaluation

Train or evaluate AGNN with a config file from `config/`:

```bash
python main_gnn.py --dataset_root <dataset_root> --config config/5way_5shot_resnet12_custom.py --mode train
python main_gnn.py --dataset_root <dataset_root> --config config/5way_5shot_resnet12_custom.py --mode eval
```

You can also pass `--pretrain_path` to initialize the backbone from a pretrained checkpoint produced by `pretrain.py`.

## Backbone Pretraining

Use `pretrain.py` to train a backbone on a standard classification dataset before running AGNN:

```bash
python pretrain.py ^
    --dataset_root <dataset_root> ^
    --split_path <split.json> ^
    --checkpoint_dir ./pretrain_checkpoints ^
    --log_dir ./pretrain_logs ^
    --num_epochs 100 ^
    --batch_size 64 ^
    --lr 1e-3
```

The script saves a backbone checkpoint that can be reused by `main_gnn.py` through `--pretrain_path`.

## Inference

`inference.py` supports hybrid open/closed-world inference. At least one of these inputs is required:

- `--support_dir`: a folder of novel class support images
- `--base_prototypes`: a `.pth` file with precomputed base prototypes

Example:

```bash
python inference.py --config config/5way_5shot_resnet12_custom.py --checkpoint <checkpoint_path> --query_dir <query_dir> --support_dir <support_dir>
```

## Streamlit Demo

The interactive demo is launched with:

```bash
run_app.bat
```

or directly:

```bash
python -m streamlit run app.py --server.headless true
```

The demo lets you manage the support set, add new images/classes, remove samples, and run predictions on query images.

## Citation

If you use this repository in research, please cite the original paper:

```bibtex
@article{cheng2023graph,
    title={Graph Neural Networks With Triple Attention for Few-Shot Learning},
    author={Cheng, Hao and Zhou, Joey Tianyi and Tay, Wee Peng and Wen, Bihan},
    journal={IEEE Transactions on Multimedia},
    year={2023},
    publisher={IEEE}
}
```

## Acknowledgment

This repository reuses ideas and components from:

- [Few-shot GNN](https://github.com/vgsatorras/few-shot-gnn)
- [Transductive Propagation Network](https://github.com/csyanbin/TPN)
- [Few-shot Meta-Baseline](https://github.com/yinboc/few-shot-meta-baseline)
- [DPGN: Distribution Propagation Graph Network for Few-shot Learning](https://github.com/megvii-research/DPGN)
- [FEAT](https://github.com/Sha-Lab/FEAT)
