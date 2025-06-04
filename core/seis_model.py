import numpy as np
import cupy as cp
import kaggle_support as kgs
import seis_forward2
import seis_prior
import seis_invert
import seis_nn
import scipy
import copy
from dataclasses import dataclass, field, fields
import matplotlib.pyplot as plt

@dataclass
class ModelOnlyFlatVel(kgs.Model):
    model: kgs.Model = field(init=True, default_factory = kgs.Model)

    def _train(self, train_data, validation_data):
        self.model.train(train_data, validation_data)

    def _infer_single(self, data):
        kpi = kgs.rms(data.seismogram.data[0,...] - cp.flip(data.seismogram.data[4,...],axis=1))
        if kpi<1e-4:
            # FlatVel
            data = self.model.infer([data])[0]
        else:
            data.do_not_cache=True
            pass
        return data

def default_model():
    model = kgs.ChainedModel()    
    model.models.append(seis_nn.default_pretrained)
    model.models.append(ModelOnlyFlatVel(model=seis_invert.InversionModel(prior=seis_prior.RowTotalVariation())))
    model.models[-1].model.cache_name = 'FlatVel'
    model.models[-1].model.write_cache = True
    model.models[-1].model.read_cache = True
    model.cache_name = 'Default'
    model.write_cache = True
    model.read_cache = True
    model.models[-1].run_in_parallel = False
    model.train([], []) # dummy

    return model

def submission_model():
    model = default_model()
    model.models = model.models[0:1]

    return model

def check_model_accuracy(model, subsample):
    model = copy.deepcopy(model)
    model.read_cache=False
    data=kgs.load_all_train_data(validation_only=True)[::subsample]
    data_out = model.infer(data)
    _,_,scores = kgs.score_metric(data_out)
    for d,score in zip(data_out,scores):
        success = True
        if 'FlatVel' in d.family:
            success = score<1.
        else: 
            success = score<200.
        if not success:            
            d.load_to_memory()
            _,ax = plt.subplots(1,2,figsize=(12,6))
            plt.sca(ax[0])
            plt.imshow(d.velocity_guess.data - cp.asnumpy(d.velocity.data))
            plt.colorbar()
            plt.sca(ax[1])
            plt.imshow(cp.asnumpy(d.velocity.data))
            plt.colorbar()
            plt.suptitle(d.cache_name())
            d.unload()
                