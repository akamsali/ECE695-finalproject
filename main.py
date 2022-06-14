from cmath import log
import torch
from datetime import datetime
import os
import pandas as pd

from nlb_tools.make_tensors import make_eval_target_tensors
from nlb_tools.evaluation import evaluate

from dataloader import get_data
from runner import NLBRunner
from model import VRNN



''' 
    load data
    dataset_name: small, medium and large
    data_path: path to the mc maze data
    bin_size: window size for spike binning
'''

dataset_name = 'mc_maze_large'
data_path = '/scratch/gilbreth/akamsali/Research/Makin/000138/sub-Jenkins'
bin_size = 5
epochs = 10000


# Extract data
phase = 'val'
dataset, train_dict, eval_dict, training_input, training_output, eval_input = get_data(data_path, dataset_name, phase, bin_size)

# Train/val split and convert to Torch tensors
num_train = int(round(training_input.shape[0] * 0.75))
train_input = torch.Tensor(training_input[:num_train])
train_output = torch.Tensor(training_output[:num_train])
val_input = torch.Tensor(training_input[num_train:])
val_output = torch.Tensor(training_output[num_train:])
eval_input = torch.Tensor(eval_input)


# creating data checkpoints
RUN_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_model'
RUN_DIR = './runs/'
if not os.path.isdir(RUN_DIR):
    os.mkdir(RUN_DIR)

# model paramters
DROPOUT = 0
L2_WEIGHT = 5e-7
LR_INIT = 1.5e-2
CD_RATIO = 0.27
HIDDEN_DIM = 40
USE_GPU = True
MAX_GPUS = 1

# initiate NLBRunner from NLB toolkit
runner = NLBRunner(
    model_init=VRNN,
    model_cfg={'x_dim': train_input.shape[2], 'h_dim': HIDDEN_DIM, 'z_dim': HIDDEN_DIM, 'out_dim': train_output.shape[2], 'n_layers': 1, 'dropout': DROPOUT},
    data=(train_input, train_output, val_input, val_output, eval_input),
    train_cfg={'lr': LR_INIT, 'alpha': L2_WEIGHT, 'cd_ratio': CD_RATIO},
    use_gpu=USE_GPU,
    num_gpus=MAX_GPUS,
)


model_dir = os.path.join(RUN_DIR, RUN_NAME)
os.mkdir(os.path.join(RUN_DIR, RUN_NAME))
train_log = runner.train(n_iter=epochs, patience=1000, save_path=os.path.join(model_dir, 'model.ckpt'), verbose=False, log_frequency=1)

# Save results
train_log = pd.DataFrame(train_log)
train_log.to_csv(os.path.join(model_dir, 'train_log.csv'))

# =============================================
# evaluate model from checkpoint after training
# =============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

conf = {'x_dim': train_input.shape[2], 'h_dim': HIDDEN_DIM, 'z_dim': HIDDEN_DIM, 'out_dim': train_output.shape[2], 'n_layers': 1, 'dropout': DROPOUT}
saved_model = VRNN(**conf).to(device)
ckpt = torch.load(f'{model_dir}/model.ckpt')
saved_model.load_state_dict(ckpt['state_dict'])

saved_model.eval()
training_input = torch.Tensor(training_input).to(device)
eval_input = eval_input.to(device)
training_predictions, _, _ = saved_model(training_input)
eval_predictions, _, _ = saved_model(eval_input)

training_predictions = training_predictions.cpu().detach().numpy()
eval_predictions = eval_predictions.cpu().detach().numpy()

tlen = train_dict['train_spikes_heldin'].shape[1]
num_heldin = train_dict['train_spikes_heldin'].shape[2]

# create submission dict for evaluation
submission = {
    'mc_maze_large': {
        'train_rates_heldin': training_predictions[:, :tlen, :num_heldin],
        'train_rates_heldout': training_predictions[:, :tlen, num_heldin:],
        'eval_rates_heldin': eval_predictions[:, :tlen, :num_heldin],
        'eval_rates_heldout': eval_predictions[:, :tlen, num_heldin:],
        'eval_rates_heldin_forward': eval_predictions[:, tlen:, :num_heldin],
        'eval_rates_heldout_forward': eval_predictions[:, tlen:, num_heldin:]
    }
}

target_dict = make_eval_target_tensors(dataset=dataset, 
                                       dataset_name='mc_maze_large',
                                       train_trial_split='train',
                                       eval_trial_split='val',
                                       include_psth=True,
                                       save_file=False)

eval_results = evaluate(target_dict, submission)

# print(eval_results[0])

import json
# eval_log = pd.DataFrame(eval_results[0])
with open(os.path.join(model_dir, 'eval_log.json'), 'w') as f:
    json.dump(eval_results[0], f)

# loss and eval plot
log_path = os.path.join(model_dir, 'train_log.csv')
train_df = pd.read_csv(log_path)
train_loss = train_df['train_cosmooth_nll']
val_loss = train_df['val_cosmooth_nll']

import matplotlib.pyplot as plt 

plt.plot(train_loss, label='train_loss')
plt.plot(val_loss, label='val_loss')
plt.xlabel("Epochs")
plt.ylabel("Poisson NLL Loss")
plt.title("Train Val Co-smooth loss")
plt.legend()
plt.savefig(model_dir+'/train_val_plot.png')