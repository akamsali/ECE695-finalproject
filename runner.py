import torch

class NLBRunner:
    """Class that handles training NLBRNN"""
    def __init__(self, model_init, model_cfg, data, train_cfg, use_gpu=False, num_gpus=1):
        self.model = model_init(**model_cfg)
        self.data = data
        if use_gpu and torch.cuda.is_available():
            device = torch.device('cuda:0')
            self.model.to(device)
            self.data = tuple([d.to(device) for d in self.data])
        self.cd_ratio = train_cfg.get('cd_ratio', 0.2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=train_cfg.get('lr', 1e-3), 
                                          weight_decay=train_cfg.get('alpha', 0.0))
    
    def make_cd_mask(self, train_input, train_output):
        """Creates boolean mask for coordinated dropout.

        In coordinated dropout, a random set of inputs is zeroed out,
        and only the corresponding outputs (i.e. same trial, timestep, and neuron)
        are used to compute loss and update model weights. This prevents
        exact spike times from being directly passed through the model.
        """
        cd_ratio = self.cd_ratio
        input_mask = torch.zeros((train_input.shape[0] * train_input.shape[1] * train_input.shape[2]), dtype=torch.bool)
        idxs = torch.randperm(input_mask.shape[0])[:int(round(cd_ratio * input_mask.shape[0]))]
        input_mask[idxs] = True
        input_mask = input_mask.view((train_input.shape[0], train_input.shape[1], train_input.shape[2]))
        output_mask = torch.ones(train_output.shape, dtype=torch.bool)
        output_mask[:, :, :input_mask.shape[2]] = input_mask
        return input_mask, output_mask
    
    def train_epoch(self):
        """Trains model for one epoch. 
        This simple script does not support splitting training samples into batches.
        """
        self.model.train()
        self.optimizer.zero_grad()
        # create mask for coordinated dropout
        train_input, train_output, val_input, val_output, *_ = self.data
        input_mask, output_mask = self.make_cd_mask(train_input, train_output)
        # mask inputs
        masked_train_input = train_input.clone()
        masked_train_input[input_mask] = 0.0
        # masked_train_input = masked_train_input.transpose(0, 1)
        # train_predictions, kld_loss, nll_loss, a , b = self.model(masked_train_input)
        train_predictions, kld_loss, nll_loss = self.model(masked_train_input)
        
        # print("model",kld_loss.shape, nll_loss.shape)
        # learn only from masked inputs
        # train_predictions = train_predictions.transpose(0,1)
        # print("train_pred_Shape:", train_predictions.shape, train_output.shape, output_mask.shape)
        p_loss = torch.nn.functional.poisson_nll_loss(train_predictions[output_mask], train_output[output_mask], log_input=False)
        # print("loss",kld_loss, nll_loss, p_loss)
        loss = p_loss + kld_loss + nll_loss
        loss.backward()
        self.optimizer.step()
        # get validation score
        train_res, train_output = self.score(train_input, train_output, prefix='train')
        val_res, val_output = self.score(val_input, val_output, prefix='val')
        res = train_res.copy()
        res.update(val_res)
        return res, (train_output, val_output)
    
    def score(self, input, output, prefix='val'):
        """Evaluates model performance on given data"""
        self.model.eval()
        predictions, pred_kld, pred_nll = self.model(input)
        # print(predictions.shape, output.shape)
        self.model.train()
        loss = torch.nn.functional.poisson_nll_loss(predictions, output, log_input=False) + pred_kld + pred_nll
        num_heldout = output.shape[2] - input.shape[2]
        cosmooth_loss = torch.nn.functional.poisson_nll_loss(
            predictions[:, :, -num_heldout:], output[:, :, -num_heldout:], log_input=False)
        return {f'{prefix}_nll': loss.item(), f'{prefix}_cosmooth_nll': cosmooth_loss.item()}, predictions

    def train(self, n_iter=1000, patience=200, save_path=None, verbose=False, log_frequency=50):
        """Trains model for given number of iterations with early stopping"""
        train_log = []
        best_score = 1e8
        last_improv = -1
        for i in range(n_iter):
            res, output = self.train_epoch()
            res['iter'] = i
            train_log.append(res)
            if verbose:
                if (i % log_frequency) == 0:
                    print(res)
            if res['val_nll'] < best_score:
                best_score = res['val_nll']
                last_improv = i
                data = res.copy()
                if save_path is not None:
                    self.save_checkpoint(save_path, data)
            if (i - last_improv) > patience:
                break
        return train_log
    
    def save_checkpoint(self, file_path, data):
        default_ckpt = {
            "state_dict": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
        }
        assert "state_dict" not in data
        assert "optim_state" not in data
        default_ckpt.update(data)
        torch.save(default_ckpt, file_path)