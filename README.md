# cs6220-project
This project contains four pretrained models on `ChestXpert` dataset. Each subfolder contains individual model's `README` instrcution to run each.

You can also see our [PPT report](https://docs.google.com/presentation/d/15Ph8bf1RSLJIzPcxWmzxRwDHENnUHQ9OQ0KCfVck9is/edit?usp=sharing)
## Setup
1. `python3 -m venv cs6200` - creates a virtual environment
2. `source cs6200/bin/activate` - activates the env
3. `pip install -r requirements.txt` - install all dependencies
4. Download a subsampled ChestXpert dataset, which originally has ~410GB, with same amount of data in ~11GB through
    1. Kaggle `#!/bin/bash
    curl -L -o ~/Downloads/chexpert-v10-small.zip\
    https://www.kaggle.com/api/v1/datasets/download/willarevalo/chexpert-v10-small` - this download the dataset into your local directory instead of being cached in huggingface.

## How to request PACE GPU
1. `salloc --gres=gpu:H100:1 --ntasks-per-node=1 --time 01:00:00` - submit a request for one node with `H100` GPU and one task per node
    - adjust the request time for your need