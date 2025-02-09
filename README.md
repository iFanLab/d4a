# D4A: An Efficient and Effective Defense Across Agnostic Adversarial Attacks

This repository provides the official implementation of **D4A**, a defense mechanism introduced in our paper *"[D4A: An Efficient and Effective Defense Across Agnostic Adversarial Attacks](https://www.sciencedirect.com/science/article/pii/S0893608024008670)"*. D4A enhances the robustness of Graph Neural Networks (GNNs) against structural adversarial attacks through **smooth-less message passing** and a **posterior distribution shift constraint**.

## Key Contributions

1. **Smooth-less Message Passing (SLMP)**: Removes graph smoothing to improve GNN robustness against structural perturbations and mitigate the "unfitting" issue.
2. **Distribution Shift Constraint (DSC)**: Reduces posterior distribution shifts induced by adversarial attacks.
3. **Extensive Evaluation**: Tested on six datasets against four attacks, demonstrating an optimal balance of accuracy, robustness, and scalability, outperforming 11 baselines overall.

## Repository Structure

```bash
d4a/
├── ckpt/           # Model checkpoints
├── configs/        # Experimental hyperparameter configs
├── data/           # Datasets
├── datasets.py     # Data loading and preprocessing module
├── layers.py       # SLMP module 
├── main.py         # Main script for training and evaluation
├── models.py       # D4A model
├── utils.py        # DSC module and other utility functions
├── env.yaml        # Conda environment config
└── train.sh        # Script for running experiments and reproducing results
```

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:iFanLab/d4a.git
   cd d4a
   ```
2. Set up the environment:
    ```bash
    conda env create -f environment.yaml
    ```

## Usage

To reproduce the results from our paper, simply execute:
```bash
bash train.sh
```

## Citation

If you find our work useful, please consider citing our paper:
```bibtex
@article{d4a2025,
title = {D4A: An efficient and effective defense across agnostic adversarial attacks},
journal = {Neural Networks},
volume = {183},
pages = {106938},
year = {2025},
author = {Xianxian Li and Zeming Gan and Yan Bai and Linlin Su and De Li and Jinyan Wang},
}
```
