# OGB_link_prediction

The structure of the code is borrowed and modified from the implementation of Pairwise Learning for Neural Link Prediction (PLNLP) framework by [zhitao-wang](https://github.com/zhitao-wang/PLNLP/)


**Example Usage:**
```
from dataset import load_dataset
from model import Model, run_link_prediction
from layer import GCN
from logger import Logger

data, split_edge, evaluator = load_dataset('ogbl-ppa')  # Dataset name

model = Model(data,
              GCN, 256, 3,      # GNN model to be used
              'MLP', 256, 3)    # Predictor to be used
logger = Logger
eval_metric = 'hits'            # Evaluation metric based on type of task

run_link_prediction(data, split_edge, model, evaluator, logger, eval_metric)
```
