# AI3603 Billiards - AI Agent Training & Evaluation

## Project Overview

This project implements an AI-driven billiards game environment where intelligent agents learn to make optimal shot decisions. The project uses realistic physics simulation, combined with reinforcement learning to train and evaluate agents capable of competing in 8-ball billiards. The main goal is to develop agents that can achieve high win rates against baseline opponents.


## Environment Setup

```bash
git clone https://github.com/xilinyu716/AI3603-Final
conda create -n billiard-ai python=3.13
conda activate billiard-ai
git clone https://github.com/SJTU-RL2/pooltool.git
cd pooltool
pip install "poetry==2.2.1"
poetry install --with=dev,docs
pip install bayesian-optimization numpy
pip install torch

```

## Evaluation
To verify your setup is correct, run:
```bash
python -m eval.evaluate
```

## Project Structure

- `poolenv.py` - Billiards environment implementation
- `agents/` - Agent implementations (NewAgent, etc.)
- `train.py` - Training script for NewAgent
- `evaluate.py` - Evaluation script to test agent performance
- `utils.py` - Utility functions

---
