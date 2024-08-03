# nextEpoch_structure_competition

## Installation
!! First fork this repo so you can push your changes !!
```
pip install -r requirements.txt
```

## Usage
This repo is a template for the competition. You need to write your pytorch model in `src/model.py` and the training loop in `train.py`. 

### Provided code
1. Dataloaders: The function `get_dataloaders()` will construct a training and validation dataloader with the specified batch size, maximum sequence length and dataset size. Each batch is a dictionary with `reference`, `sequence`, and `structure` elements
2. Template model: The class `RNA_net` embeds the integer encoded sequence into the sequence and matrix representations, and should be completed to crete a secondary structure model
You can use some utils in `util.py` for general logging and metrics
3. Metrics: we provide functions to compute the Precision, Recall and F1 score metrics in `util.py`
4. Plotting: we provide a function to save a plot of a structure prediction compared to its ground truth in `util.py`
