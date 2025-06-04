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

def _make_default_pretrained():
    models = []
    print(kgs.brendan_model_dir+"*.pth")
    for f in sorted(glob.glob(kgs.brendan_model_dir+"*caformer*.pt")):
        print("Loading: ", f)
        m = Net(
            backbone="caformer_b36.sail_in22k_ft_in1k",
            pretrained=False,
        )
        state_dict= torch.load(f, map_location='cpu', weights_only=True)
        state_dict= {k.removeprefix("_orig_mod."):v for k,v in state_dict.items()} # Remove torch.compile() prefix

        m.load_state_dict(state_dict)
        models.append(m)

    assert len(models)==1
    
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

    batch_size = 16

    def _train(self, train_data, validation_data):
        raise Exception('Not supported')

    def _infer(self,data):
        cpu,device = kgs.prep_pytorch(0, False, False)
        self.model.to('cpu')
        # if torch.cuda.device_count() > 1:
        #     print(f"Found {torch.cuda.device_count()} GPUs, using DataParallel")           
        #     self.model = torch.nn.DataParallel(self.model)
        self.model.to(device)   # moves the (possibly wrapped) model to cuda:0
        
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
                #data[i].velocity_guess.data = cp.from_dlpack(torch.utils.dlpack.to_dlpack(output[i_sub,0,...]))
                data[i].velocity_guess.data = output[i_sub,0,...].cpu().numpy()
                data[i].velocity_guess.min_vel = cp.min(data[i].velocity_guess.data)

        self.model.to('cpu')
        return data

    # def _infer(self, data):

    #     print('hi')

    #     kgs.prep_pytorch(0, True, False)
        
    #     # Detect available GPUs
    #     n_gpus = torch.cuda.device_count()
    #     print(n_gpus)
    #     primary_device = torch.device('cuda:0' if n_gpus > 0 else 'cpu')
    #     print(primary_device)

    #     # Ensure model is on CPU before DataParallel wrapping
    #     self.model.cpu()

    #     # Wrap model for multi-GPU if more than one GPU is available
    #     if n_gpus > 1:
    #         print(f"Using {n_gpus} GPUs for inference")
    #         device_ids = list(range(n_gpus))
    #         self.model = torch.nn.DataParallel(
    #             self.model,
    #             device_ids=device_ids,
    #             output_device=device_ids[0]
    #         )

    #     # Move (wrapped) model to primary GPU
    #     self.model.to(primary_device)
    #     self.model.eval()

    #     # Split data indices into batches
    #     sub_list = np.array_split(
    #         np.arange(len(data)),
    #         len(data) // self.batch_size + 1
    #     )

    #     for inds in tqdm(sub_list):
    #         if not len(inds):
    #             continue

    #         # Allocate input tensor on the primary device
    #         x = torch.empty(
    #             (len(inds), 5, 1000, 70),
    #             dtype=torch.float32,
    #             device=primary_device
    #         )

    #         # Fill the batch tensor via DLPack zero-copy from CuPy
    #         for i_sub, i in enumerate(inds):
    #             d = data[i]
    #             d.seismogram.load_to_memory(load_last_row=True)
    #             dlpack = d.seismogram.data.astype(cp.float32).toDlpack()
    #             x[i_sub] = torch.utils.dlpack.from_dlpack(dlpack)
    #             d.seismogram.unload()

    #         # Inference (autocast for mixed precision)
    #         with torch.no_grad(), autocast(device_type='cuda'):
    #             output = self.model(x)

    #         # Copy results back to numpy and assign
    #         for i_sub, i in enumerate(inds):
    #             d = data[i]
    #             d.velocity_guess = kgs.Velocity()
    #             arr = output[i_sub, 0].cpu().numpy()
    #             d.velocity_guess.data = arr
    #             d.velocity_guess.min_vel = arr.min()

    #     # Move model back to CPU to free GPU memory
    #     self.model.cpu()
    #     kgs.claim_gpu('')
    #     return data

default_pretrained = _make_default_pretrained()