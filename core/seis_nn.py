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
#import torch.utils.dlpack

def make_default_pretrained():
    models = []
    print(kgs.brendan_model_dir+"*.pth")
    for f in sorted(glob.glob(kgs.brendan_model_dir+"*.pth")):
        print("Loading: ", f)
        m = Net(
            backbone="convnext_small.fb_in22k_ft_in1k",
            pretrained=False,
        )
        state_dict= torch.load(f, map_location='cpu', weights_only=True)
        state_dict= {k.removeprefix("_orig_mod."):v for k,v in state_dict.items()} # Remove torch.compile() prefix
    
        m.load_state_dict(state_dict)
        models.append(m)
    
    # Combine
    model = EnsembleModel(models)
    model = model.to('cpu')
    model = model.eval()
    print("n_models: {:_}".format(len(models)))

    model_out = NeuralNetwork()
    model_out.model = model
    model_out.state = 1
    model_out.run_in_parallel = False

    model_out.write_cache=True
    model_out.read_cache=True
    model_out.cache_name = 'brendan'
    return model_out

class NeuralNetwork(kgs.Model):

    model = 0
    model_prepped = False

    batch_size = 100

    def _train(train_data, validation_data):
        raise Exception('Not supported')

    def _infer(self,data):
        cpu,device = kgs.prep_pytorch(0, True, False)
        if not self.model_prepped:
            self.model = self.model.to(device)
            self.model_prepped = True
        
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
                data[i].velocity_guess.data = cp.from_dlpack(torch.utils.dlpack.to_dlpack(output[i_sub,0,...]))
                data[i].velocity_guess.min_vel = cp.min(data[i].velocity_guess.data)

        return data