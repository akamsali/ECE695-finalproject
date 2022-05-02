from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors

import numpy as np

def get_data(data_path, dataset_name, phase='test', bin_size=5):
    """Function that extracts and formats data for training model"""
    # data_path = '/scratch/gilbreth/akamsali/Research/Makin/000138/sub-Jenkins'

    dataset = NWBDataset(data_path, 
        skip_fields=['cursor_pos', 'eye_pos', 'cursor_vel', 'eye_vel', 'hand_pos'])
    dataset.resample(5)
    train_split = ['train', 'val'] if phase == 'test' else 'train'
    eval_split = phase
    train_dict = make_train_input_tensors(dataset, dataset_name, train_split, save_file=False, include_forward_pred=True)
    eval_dict = make_eval_input_tensors(dataset, dataset_name, eval_split, save_file=False)
    training_input = np.concatenate([
        train_dict['train_spikes_heldin'],
        np.zeros(train_dict['train_spikes_heldin_forward'].shape),
    ], axis=1)
    training_output = np.concatenate([
        np.concatenate([
            train_dict['train_spikes_heldin'],
            train_dict['train_spikes_heldin_forward'],
        ], axis=1),
        np.concatenate([
            train_dict['train_spikes_heldout'],
            train_dict['train_spikes_heldout_forward'],
        ], axis=1),
    ], axis=2)
    eval_input = np.concatenate([
        eval_dict['eval_spikes_heldin'],
        np.zeros((
            eval_dict['eval_spikes_heldin'].shape[0],
            train_dict['train_spikes_heldin_forward'].shape[1],
            eval_dict['eval_spikes_heldin'].shape[2]
        )),
    ], axis=1)
    # del dataset
    return dataset, train_dict, eval_dict, training_input, training_output, eval_input