# ECE695-finalproject

## Data 
Dataset: MC_Maze dataset from Shenoy's Lab

To download we use ```dandi```. Install and download dataset as follows:

```
pip install dandi
```
For download, go to [MC_maze](https://neurallatents.github.io/datasets#mcmaze) and copy link of required size from dandi repo and download as follows
```
dandi download <link>
```

## Installation
```
pip install git+https://github.com/neurallatents/nlb_tools.git
pip install torch
pip install evalai
```
## Run
To run, change the `data_path`, `dataset_name` and epochs  in `main.py` and run the file.
This does the following:
1. Train the model on the given data set and create checkpoint
2. Create log files with train and validation loss
3. Evaluate the checkpoint and save the results in eval_log.csv
4. Generate and save loss plots
