import numpy as np
import cupy as cp
import kaggle_support as kgs
import seis_forward2
import seis_prior
import seis_invert
import seis_numerics
import seis_nn
import copy
from dataclasses import dataclass, field, fields

test_mode = False # if True makes faster models

# Model for FlatVelA and FlatVelB
def model_FlatVel():
    # This is a single LBFGS step
    model = seis_invert.InversionModel()
    # 1D total variation prior
    model.prior = seis_prior.RowTotalVariation()
    model.prior.λ = 1e-8
    model.prior.epsilon = 0.1
    model.maxiter = 1000 if not test_mode else 30
    model.lbfgs_tolerance_change = 1e-9
    model.lbfgs_tolerance_grad = 1e-7
    model.round_results = True # round results to nearest integer
    return model

# Model for StyleA
def model_Style_A():   

    model_full = kgs.ChainedModel() # 2 step model

    # Step 1 stops based on a gradient condition
    model = seis_invert.InversionModel()
    model.maxiter = 2500 if not test_mode else 50
    # Gaussian Process prior
    model.prior = seis_prior.SquaredExponential()
    model.prior.transform = True
    model.prior.svd_cutoff = 1.
    model.prior.λ = 10**-12
    model.lbfgs_tolerance_grad = 10**6.5

    # Step 2 does a single Gauss-Newton pass
    model2 = copy.deepcopy(model)
    model2.maxiter = 0
    model2.cache_name = 'Style_A_refine'

    # Combine
    model_full.models = [model,model2]
    if test_mode:
        model2.prior.svd_cutoff = 10000.    
    return model_full

# Model for StyleB
def model_Style_B():

    model_full = kgs.ChainedModel() # 2 step model

    # Step 1 runs for 1500 iterations
    model = seis_invert.InversionModel()
    model.maxiter = 1500 if not test_mode else 50
    # Gaussian Process prior  
    model.prior = seis_prior.SquaredExponential()
    model.prior.length_scale = 1.96
    model.prior.sigma = 158
    model.prior.noise = 58
    model.prior.sigma_mean = 1239
    model.prior.sigma_slope = 92
    model.prior.transform = False
    model.prior.λ = 10**-15

    # Step 2 stops when the MSE error for the target seismogram is under 2.5e-9
    model2 = copy.deepcopy(model)
    model2.maxiter = 7500 if not test_mode else 10
    model2.seis_error_tolerance = 2.5e-9

    # Combine    
    model_full.models = [model,model2]
    return model_full

# Model for some 'easy' fault datasets, specifically those with only 2 velocities (except around the fault)
def model_TV2Deasy():
    # This is a single LBFGS step
    model = model_TV2D().models[0] # take over the model for harder datasets
    # Adaptations for these easy datasets
    model.prior.λ = 10**-7
    model.lbfgs_tolerance_grad *= 10**1
    model.maxiter = 600
    model.round_results = True # round results to nearest integer
    return model

# Model used for all other datasets
def model_TV2D():
    model_full = kgs.ChainedModel() # 3 step model

    # Step 1: LBFGS with gradient stopping criterion
    model = seis_invert.InversionModel()
    # 2D total variation prior
    model.prior = seis_prior.TotalVariation()
    model.prior.λ = 10**-8
    model.maxiter = 2500 if not test_mode else 75    
    model.lbfgs_tolerance_grad = 10**3.5

    # Step 2: Same as step 1, but allowed to run longer
    # NOTE: I think just starting with step 2 is better; current choice is due to legacy.
    model2 = copy.deepcopy(model)
    model2.maxiter = 10000 if not test_mode else 5

    # Step 3: LBFGS step with more strict prior. 
    model3 = copy.deepcopy(model)
    model3.maxiter = 1500 if not test_mode else 10
    model3.prior.λ /= 10 # more strict prior
    model3.lbfgs_tolerance_grad = 10**2.5
    # Single basis function for areas with similar velocity
    old_prior = model3.prior
    model3.prior = seis_prior.RestrictFlatAreas()
    model3.prior.underlying_prior = old_prior
    model3.prior.diff_threshold1 = 1.
    model3.prior.rrange = 1

    # Combine    
    model_full.models = [model,model2,model3]
    model_full.round_results = True # round results to nearest integer
    return model_full


# Dispatches to one of five models based on classification of the velocity profile in the initial guess.
@dataclass
class ModelSplit(kgs.Model):
    model_FlatVel : kgs.Model = field(init=True, default_factory = model_FlatVel )
    model_Style_A : kgs.Model = field(init=True, default_factory = model_Style_A )
    model_Style_B : kgs.Model = field(init=True, default_factory = model_Style_B )
    model_TV2D    : kgs.Model = field(init=True, default_factory = model_TV2D    )
    model_TV2Deasy: kgs.Model = field(init=True, default_factory = model_TV2Deasy)

    P_identify_style_A = 0 # used in classifying styleA

    def _train(self, train_data, validation_data):
        # Find prior precision matrix for styleA
        prior_style_A = copy.deepcopy(self.model_Style_A.models[0].prior)
        prior_style_A.prepped = False
        prior_style_A.transform = False
        prior_style_A.prep()
        self.P_identify_style_A = prior_style_A.P

        # Train individual models
        self.model_FlatVel.train(train_data, validation_data)
        self.model_Style_A.train(train_data, validation_data)
        self.model_Style_B.train(train_data, validation_data)
        self.model_TV2D.train(train_data, validation_data)
        self.model_TV2Deasy.train(train_data, validation_data)
        

    def _infer_single(self, data):
        # Are we FlatVel? Based on symmetry of the seismogram
        kpi_FlatVel = kgs.rms(data.seismogram.data[0,...] - cp.flip(data.seismogram.data[4,...],axis=1))
        if kpi_FlatVel<1e-4:
            data = self.model_FlatVel.infer([data])[0]
        else:

            # Are we StyleA? Based on log likelihood using StyleA prior.
            vel_cp = copy.deepcopy(data.velocity_guess)
            vel_cp.to_cupy()
            vec = vel_cp.to_vector()
            kpi_Style_A = cp.asnumpy(vec[:-1,:].T@(self.P_identify_style_A@vec[:-1,:]))        
            if kpi_Style_A<np.exp(13):
                if not data.family=='test':
                    assert data.family=='Style_A'                
                data = self.model_Style_A.infer([data])[0] 
            
            # Are we StyleB? Based on size of largest flat patch in the profile (StyleB doesn't have flat patches)
            elif kpi_style_B(vel_cp.data)<95:
                if not data.family=='test':
                    assert data.family=='Style_B'   
                data = self.model_Style_B.infer([data])[0]   
            else:
                if not data.family=='test':
                    assert not 'FlatVel' in data.family and not 'Style' in data.family

                # Are there just 2 velocities in play?
                if kpi_fault_A(data.velocity_guess.data)>4100:
                    data = self.model_TV2Deasy.infer([data])[0]                       
                else:                

                    # Run the model for 'all other' datasets.
                    data = self.model_TV2D.infer([data])[0]   
        return data

# The main model. First uses Brendan Artley's model to create an initial guess, then refines it with Bayesian FWI (above).
def default_model():
    model = kgs.ChainedModel()    
    model.models.append(seis_nn.default_pretrained) # Brendan Artley's model
    model.models.append(ModelSplit()) # Bayesian FWI

    # Cache reuslts
    model.cache_name = 'Default'
    model.write_cache = True
    model.read_cache = True

    # 'Train' the model (but none of our models really need training - Brendan's model is already pretrained)
    model.train([], [])
    return model


# Further helper functions, not commented
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