import numpy as np
import cupy as cp
import kaggle_support as kgs
import seis_forward2
import seis_prior
import seis_invert
import seis_numerics
import seis_nn
import scipy
import copy
from dataclasses import dataclass, field, fields
import matplotlib.pyplot as plt

test_mode = False # if True makes faster models

def model_FlatVel():
    model = seis_invert.InversionModel()
    model.prior = seis_prior.RowTotalVariation()
    model.prior.λ = 1e-8
    model.lambda_list = [1e-8]
    model.prior.epsilon = 0.1
    model.iter_list = [10000] if not test_mode else [30]
    model.use_new_bfgs = False
    model.round_results = True

    model.cache_name = 'FlatVel'
    model.write_cache = True
    model.read_cache = True
    return model

def model_Style_A():   

    model_full = kgs.ChainedModel()
    
    model = seis_invert.InversionModel()
    model.iter_list = [2500] if not test_mode else [50,-50]
    
    model.prior = seis_prior.SquaredExponential()
    model.prior.transform = True
    model.prior.svd_cutoff = 1.
    model.prior.λ = 10**-12
    model.lbfgs_tolerance_grad = 10**6.5

    model.cache_name = 'Style_A'
    model.write_cache = True
    model.read_cache = True

    model2 = copy.deepcopy(model)
    model2.iter_list = [0]
    model2.cache_name = 'Style_A_refine'

    model_full.models = [model,model2]

    if test_mode:
        model2.prior.svd_cutoff = 10000.
    
    return model_full

def model_Style_B():

    model_full = kgs.ChainedModel()
    
    model = seis_invert.InversionModel()
    model.iter_list = [1500] if not test_mode else [50]
    #model.iter_list = [3500] if not test_mode else [50];model.lbfgs_tolerance_grad = 10**3.5
    
    
    model.prior = seis_prior.SquaredExponential()
    model.prior.length_scale = 1.96
    model.prior.sigma = 158
    model.prior.noise = 58
    model.prior.sigma_mean = 1239
    model.prior.sigma_slope = 92
    model.prior.transform = False
    model.prior.λ = 10**-15

    model.cache_name = 'Style_B'
    model.write_cache = True
    model.read_cache = True

    model2 = copy.deepcopy(model)
    model2.iter_list = [7500] if not test_mode else [10]
    model2.seis_error_tolerance = 2.5e-9
    model2.cache_name = 'Style_B_refine'
    
    model_full.models = [model,model2]

    return model_full

def model_TV2D():
    model = seis_invert.InversionModel()
    model.iter_list = [2500] if not test_mode else [75]

    model.prior = seis_prior.TotalVariation()
    model.prior.λ = 10**-8
    model.lbfgs_tolerance_grad = 10**3.5

    model.cache_name = 'TV2D'
    model.write_cache = True
    model.read_cache = True
    return model

def model_TV2D_refine():
    model = model_TV2D()
    
    #model.prior.λ = 10**-8
    #model.lbfgs_tolerance_grad = 10**2.5
    model.iter_list = [10000] if not test_mode else [5]

    model.cache_name = 'model_TV2D_refine'
    model.write_cache = True
    model.read_cache = True

    # model = DummyModel()
    # model.cache_name = 'model_TV2D_refine'
    # model.write_cache = False
    # model.read_cache = True
    # model.do_not_cache_mode = False
    return model

def model_TV2D_ultimo():
    model = model_TV2D()
    model.iter_list = [1500] if not test_mode else [10]
    model.prior.λ /= 10
    model.lbfgs_tolerance_grad = 10**2.5
    model.cache_name = 'TV2D_ultimo'
    model.write_cache = True
    model.read_cache = True
    old_prior = model.prior
    model.prior = seis_prior.RestrictFlatAreas()
    model.prior.underlying_prior = old_prior
    model.prior.diff_threshold1 = 1.
    model.prior.rrange = 1
    return model

def model_TV2Deasy():
    model = model_TV2D()
    model.prior.λ = 10**-7
    #model.models[1].model_TV2Deasy.lbfgs_tolerance_grad = 0.
    model.lbfgs_tolerance_grad *= 10**1
    #model.lbfgs_tolerance_grad = 'check'
    model.iter_list = [600]
    model.cache_name = 'TV2Deasy'
    model.write_cache = True
    model.read_cache = True
    return model

    

StyleAseen=0
StyleBseen=0
FlatVelseen= 0
@dataclass
class ModelSplit(kgs.Model):
    model_FlatVel: kgs.Model = field(init=True, default_factory = model_FlatVel)
    model_Style_A: kgs.Model = field(init=True, default_factory = model_Style_A)
    model_Style_B: kgs.Model = field(init=True, default_factory = model_Style_B)
    model_TV2D   : kgs.Model = field(init=True, default_factory = model_TV2D   )
    model_TV2D_refine: kgs.Model = field(init=True, default_factory = model_TV2D_refine   )
    model_TV2D_ultimo: kgs.Model = field(init=True, default_factory = model_TV2D_ultimo   )
    model_TV2Deasy: kgs.Model = field(init=True, default_factory = model_TV2Deasy   )

    P_identify_style_A = 0

    refine_threshold = np.inf #0. # but using dummy

    def _train(self, train_data, validation_data):
        prior_style_A = copy.deepcopy(self.model_Style_A.models[0].prior)
        prior_style_A.prepped = False
        prior_style_A.transform = False
        prior_style_A.prep()
        self.P_identify_style_A = prior_style_A.P
        
        self.model_FlatVel.train(train_data, validation_data)
        self.model_Style_A.train(train_data, validation_data)
        self.model_Style_B.train(train_data, validation_data)
        self.model_TV2D.train(train_data, validation_data)
        self.model_TV2D_refine.train(train_data, validation_data)
        self.model_TV2Deasy.train(train_data, validation_data)
        self.model_TV2D_ultimo.train(train_data, validation_data)
        

    def _infer_single(self, data):
        global StyleAseen, StyleBseen, FlatVelseen
        kpi_FlatVel = kgs.rms(data.seismogram.data[0,...] - cp.flip(data.seismogram.data[4,...],axis=1))
        if kpi_FlatVel<1e-4:
            # FlatVel
            data = self.model_FlatVel.infer([data])[0]
            FlatVelseen+=1
        else:
            vel_cp = copy.deepcopy(data.velocity_guess)
            vel_cp.to_cupy()
            vec = vel_cp.to_vector()
            kpi_Style_A = cp.asnumpy(vec[:-1,:].T@(self.P_identify_style_A@vec[:-1,:]))
            #print(kpi_Style_A)            
            if kpi_Style_A<np.exp(13):
                if not data.family=='test':
                    assert data.family=='Style_A'                
                data = self.model_Style_A.infer([data])[0] 
                StyleAseen+=1
            elif kpi_style_B(vel_cp.data)<95:
                if not data.family=='test':
                    assert data.family=='Style_B'   
                data = self.model_Style_B.infer([data])[0]   
                StyleBseen+=1
            else:
                if not data.family=='test':
                    assert not 'FlatVel' in data.family and not 'Style' in data.family
                if kpi_fault_A(data.velocity_guess.data)>4100:
                    # print('Skipped an easy TV2D')
                    # probably FlatFault_A or CurveFault_A; these are really good already
                    data = self.model_TV2Deasy.infer([data])[0]                       
                else:                
                    data = self.model_TV2D.infer([data])[0]   
                    
                    # vel = copy.deepcopy(data.velocity_guess)
                    # vel.to_cupy()
                    # vel.data = vel.data.astype(kgs.base_type_gpu)
                    # vel.min_vel = vel.min_vel.astype(kgs.base_type_gpu)
                    # seis = kgs.Seismogram()
                    # seis.from_vector(seis_forward2.vel_to_seis(vel.to_vector())[0])     
                    # data.seismogram.load_to_memory()
                    # seis_err_rms_before = kgs.rms(seis.to_vector() - data.seismogram.to_vector()).get()
                    # #print(seis_err_rms_before)
                    # if seis_err_rms_before>self.refine_threshold:
                    data = self.model_TV2D_refine.infer([data])[0]   
                    data = self.model_TV2D_ultimo.infer([data])[0]   
                        # vel = copy.deepcopy(data.velocity_guess)
                        # vel.to_cupy()
                        # vel.data = vel.data.astype(kgs.base_type_gpu)
                        # vel.min_vel = vel.min_vel.astype(kgs.base_type_gpu)
                        # seis = kgs.Seismogram()
                        # seis.from_vector(seis_forward2.vel_to_seis(vel.to_vector())[0])                    
                        # data.seismogram.load_to_memory()
                        # seis_err_rms_after = kgs.rms(seis.to_vector() - data.seismogram.to_vector()).get()
                        # print(seis_err_rms_before, seis_err_rms_after)
                # round
                data.velocity_guess.data = np.round(data.velocity_guess.data)
                data.velocity_guess.min_vel = np.round(data.velocity_guess.min_vel)
        return data

@dataclass
class DummyModel(kgs.Model):
    do_not_cache_mode = True
    def _infer_single(self, data):
        if self.do_not_cache_mode:
            data.do_not_cache = True
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

def kpi_style_B(vel):
    vals = np.round(vel/3)
    _,counts = cp.unique(vals,return_counts=True)
    return cp.max(counts).get()

def kpi_fault_A(mat):
    labels,count = seis_numerics.label_thresholded_components(mat, 10, connectivity=4)
    _,counts = np.unique(labels, return_counts=True)
    counts=np.sort(counts)
    if len(counts)==1:
        return 4900
    else:
        return counts[-1]+counts[-2]

def submission_model():
    model = default_model()
    model.models = model.models[0:1]
    model.write_cache = False
    return model

def check_model_accuracy(model, subsample):
    model = copy.deepcopy(model)
    kgs.disable_caching = True
    data=kgs.load_all_train_data(validation_only=True)[::subsample]
    data_out = model.infer(data)
    _,_,scores = kgs.score_metric(data_out)
    for d,score in zip(data_out,scores):
        success = True
        if 'FlatVel' in d.family:
            success = score<1.
        elif 'Style_A' in d.family:  
            success = score<10.
        if not success:            
            d.load_to_memory()
            _,ax = plt.subplots(1,2,figsize=(12,6))
            plt.sca(ax[0])
            plt.imshow(d.velocity_guess.data - cp.asnumpy(d.velocity.data))
            plt.colorbar()
            plt.title(score)
            plt.sca(ax[1])
            plt.imshow(cp.asnumpy(d.velocity.data))
            plt.colorbar()
            plt.suptitle(d.cache_name())
            d.unload()