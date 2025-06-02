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
            pass
        return data

def default_model():
    model = kgs.ChainedModel()    
    model.models.append(seis_nn.make_default_pretrained())
    model.models.append(ModelOnlyFlatVel(model=seis_invert.InversionModel(prior=seis_prior.RowTotalVariation())))
    model.models[-1].model.cache_name = 'FlatVel'
    model.models[-1].model.write_cache = True
    model.models[-1].model.read_cache = True
    model.models[-1].run_in_parallel = True
    model.train([], []) # dummy

    return model
