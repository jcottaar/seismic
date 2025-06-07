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

def model_FlatVel():
    model = seis_invert.InversionModel()
    model.prior = seis_prior.RowTotalVariation()
    model.prior.λ = 1e-8
    model.prior.epsilon = 0.1

    model.cache_name = 'FlatVel'
    model.write_cache = True
    model.read_cache = True
    return model

def model_Style_A():
    model = seis_invert.InversionModel()
    model.maxiter = 10
    print('too low')
    
    model.prior = seis_prior.SquaredExponential()
    model.prior.transform = True
    model.prior.svd_cutoff = 1.
    model.prior.λ = 10**-11
    model.do_gn=False

    model.cache_name = 'Style_A'
    model.write_cache = True
    model.read_cache = True
    return model

@dataclass
class ModelSplit(kgs.Model):
    model_FlatVel: kgs.Model = field(init=True, default_factory = model_FlatVel)
    model_Style_A: kgs.Model = field(init=True, default_factory = model_Style_A)

    P_identify_style_A = 0

    def _train(self, train_data, validation_data):
        self.model_FlatVel.train(train_data, validation_data)
        self.model_Style_A.train(train_data, validation_data)
        prior_style_A = copy.deepcopy(self.model_Style_A.prior)
        prior_style_A.transform = False
        prior_style_A.basis_functions() # prep
        self.P_identify_style_A = prior_style_A.P

    def _infer_single(self, data):
        kpi_FlatVel = kgs.rms(data.seismogram.data[0,...] - cp.flip(data.seismogram.data[4,...],axis=1))
        if kpi_FlatVel<1e-4:
            # FlatVel
            data = self.model_FlatVel.infer([data])[0]
        else:
            vel_cp = copy.deepcopy(data.velocity_guess)
            vel_cp.to_cupy()
            vec = vel_cp.to_vector()
            kpi_Style_A = cp.asnumpy(vec[:-1,:].T@(self.P_identify_style_A@vec[:-1,:]))
            #print(kpi_Style_A)
            if kpi_Style_A<np.exp(15):
                if not data.family=='test':
                    assert data.family=='Style_A'
            else:
                if not data.family=='test':
                    assert 'FlatVel' not in data.family and not data.family=='Style_A'
            data.do_not_cache=True
            pass
        return data

def default_model():
    model = kgs.ChainedModel()    
    model.models.append(seis_nn.default_pretrained)
    model.models.append(ModelSplit())
    # model.models.append(ModelOnlyFlatVel(model=seis_invert.InversionModel(prior=seis_prior.RowTotalVariation())))
    # model.models[-1].model.cache_name = 'FlatVel'
    # model.models[-1].model.write_cache = True
    # model.models[-1].model.read_cache = True
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
    model.models[-1].model.read_cache = False
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
                