'''
Integrates Brendan Artley's model as described here into my framework: https://www.kaggle.com/code/brendanartley/caformer-full-resolution-improved
'''

import numpy as np
import cupy as cp
import kaggle_support as kgs
import copy
import importlib

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.amp import autocast
from _model import Net, EnsembleModel
import glob

def _make_default_pretrained():
    # Load pretrained model
    models = []
    for f in sorted(glob.glob(kgs.brendan_model_dir+"*caformer*.pt")):
        m = Net(
            backbone="caformer_b36.sail_in22k_ft_in1k",
            pretrained=False,
        )
        state_dict= torch.load(f, map_location='cpu', weights_only=True)
        state_dict= {k.removeprefix("_orig_mod."):v for k,v in state_dict.items()} # Remove torch.compile() prefix

        m.load_state_dict(state_dict)
        models.append(m)

    if not len(models)==1:
        raise Exception('Found no or more than one pretrained model')
    
    # Combine (legacy code when there were still multiple models)
    model = EnsembleModel(models)
    model = model.to('cpu')
    model = model.eval()

    model_out = NeuralNetwork() # defined belowed
    model_out.model = model
    model_out.state = 1 # declare ourselved trained
    model_out.run_in_parallel = False

    return model_out

class NeuralNetwork(kgs.Model):

    model = 0 # to be set by user
    batch_size = 16 # batch size durin ginference

    def _train(self, train_data, validation_data):
        raise Exception('Not supported') # only pretrained allowed

    def _infer(self,data):
        # Prep GPU
        cpu,device = kgs.prep_pytorch(0, False, False)
        self.model.to('cpu')
        self.model.to(device)   

        # Run batches
        sub_list = np.array_split(np.arange(len(data)), len(data)//self.batch_size+1)
        for inds in tqdm(sub_list):
            if len(inds)==0: continue
            x = torch.empty((len(inds),5,1000,70) , dtype=torch.float32, device=device)
            for i_sub, i in enumerate(inds):
                data[i].seismogram.load_to_memory(load_last_row=True)
                x[i_sub,...] = torch.utils.dlpack.from_dlpack(data[i].seismogram.data.toDlpack())
                data[i].seismogram.unload()
            with torch.no_grad():        
                with autocast('cuda'):
                    output = self.model(x)
            for i_sub, i in enumerate(inds):
                data[i].velocity_guess = kgs.Velocity()
                data[i].velocity_guess.data = output[i_sub,0,...].cpu().numpy()
                data[i].velocity_guess.min_vel = cp.min(data[i].velocity_guess.data)

        # Unload model
        self.model.to('cpu')
        
        return data

# Preload model
default_pretrained = _make_default_pretrained()