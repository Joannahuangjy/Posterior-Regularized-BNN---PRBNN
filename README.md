# PRBNN

PyTorch implementation of PRBNN for uncertainty-aware regression, including 1D simulation scripts, companion notebooks, and a solar energy forecasting case study with GPU support.

The implementation accompanies the PRBNN paper: [arXiv:2210.08608](https://arxiv.org/pdf/2210.08608.pdf).

## Repository Structure

```text
.
├── PRBNN_1d_regression/       # 1D regression scripts and companion notebooks
├── PRBNN_solar_case_study/    # Solar forecasting model, data utilities, and runner
├── README.md
└── requirements.txt
```

## Setup

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch following the command recommended for your system on the [official PyTorch install page](https://pytorch.org/get-started/locally/). Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1D Regression

Run the Python scripts directly:

```bash
python PRBNN_1d_regression/bbb_aleatoric_soft.py
python PRBNN_1d_regression/bbb_aleatoric_con_hard_loop.py
```

The `.ipynb` notebooks are kept as companion files for visualization and GitHub-rendered inspection. Open them with Jupyter if you want to view the original notebook flow:

```bash
jupyter notebook PRBNN_1d_regression
```

### Solar Case Study

The solar case study expects a local dataset file named `Know_regression.pkl`. The dataset is proprietary and is not included in this repository.

Place the dataset in the repository root, then run:

```bash
python -m PRBNN_solar_case_study.main
```

By default, training writes `model_checkpoint.pth` and the evaluation figure writes `results_with_hard_constraint.png`. These generated files are ignored by Git.

## Notes

- CUDA is used automatically when available; otherwise the code falls back to CPU.
- Local datasets, model checkpoints, and generated outputs are excluded via `.gitignore`.
- The dependency file is intentionally lightweight. If exact reproducibility is required, pin package versions for the machine and CUDA/PyTorch stack used in your experiments.

## License

No license has been selected yet. Add a `LICENSE` file before publishing if you want others to reuse, modify, or redistribute the code.
