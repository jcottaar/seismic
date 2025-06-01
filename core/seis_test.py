# Original source: XXX
# Python translation adapted from: https://www.kaggle.com/code/jaewook704/waveform-inversion-vel-to-seis by Jae-Wook Kim

import numpy as np
import cupy as cp
import kaggle_support as kgs
import seis_forward
import seis_prior
import seis_invert
import copy
import importlib
import matplotlib.pyplot as plt

def test_stuff_on_one_case(d, expected_match, test_reference_mode=False):

    d.load_to_memory()

    # Check vector transforms
    d.velocity.from_vector(d.velocity.to_vector())
    d.seismogram.from_vector(d.seismogram.to_vector())

    # Test matching to host's forward model
    seis_pred = seis_forward.vel_to_seis(d.velocity,d.seismogram)[0]
    mismatch = kgs.rms(seis_pred.data - d.seismogram.data)
    print(mismatch)
    assert mismatch < expected_match
    if test_reference_mode:
        #seis_forward.reference_mode = True
        #seis_pred_ref = seis_forward.vel_to_seis(d.velocity,d.seismogram)[0]
        #assert (kgs.rms(seis_pred.data - seis_pred_ref.data)/ kgs.rms(seis_pred_ref.data - d.seismogram.data))<1e-5
        #seis_forward.reference_mode = False
        seis_pred_ref = copy.deepcopy(seis_pred)
        seis_pred_ref.from_vector( seis_forward.vel_to_seis_ref(d.velocity.to_vector()) )
        print(kgs.rms(seis_pred.data - seis_pred_ref.data), kgs.rms(seis_pred.data))
        assert (kgs.rms(seis_pred.data - seis_pred_ref.data)/ kgs.rms(seis_pred.data))<1e-5
        

    # Test prep_run_diff
    N_test = 2
    offset_vecs = 1e-3*cp.array(np.random.default_rng(seed=0).normal(0,1,(N_test,len(d.velocity.to_vector()))), dtype=kgs.base_type_gpu)
    base_result = seis_forward.prep_run(d.velocity,3)
    
    finite_difference = dict()
    for ii in range(N_test):
        vel_offset = copy.deepcopy(d.velocity)    
        vel_offset.from_vector(vel_offset.to_vector() + offset_vecs[ii:ii+1,:].T)      
        offset_result = seis_forward.prep_run(vel_offset,3)
        for jj in range(len(offset_result)):
            finite_difference[ii,jj] = offset_result[jj]-base_result[jj]

    diff_result = seis_forward.prep_run_diff(d.velocity, cp.reshape(offset_vecs[:,:-1],(N_test,70,70)), offset_vecs[:,-1],3)
    for ii in range(N_test):
        for jj in range(len(offset_result)):
            assert (kgs.rms(diff_result[jj][ii,...] - finite_difference[ii,jj])/kgs.rms(finite_difference[ii,jj])) < 1e-6

    # Test vel_to_seis_diff
    N_test = 2
    offset_vecs = 1e-4*cp.array(np.random.default_rng(seed=0).normal(0,1,(len(d.velocity.to_vector()),N_test)), dtype=kgs.base_type_gpu)
    base_result, diff_result = seis_forward.vel_to_seis(d.velocity, d.seismogram, offset_vecs)
    #base_file = kgs.dill_load(kgs.temp_dir + 'nondiff')
    
    finite_difference = dict()
    #finite_difference_file = dict()
    for ii in range(N_test):
        vel_offset = copy.deepcopy(d.velocity)    
        vel_offset.from_vector(vel_offset.to_vector() + offset_vecs[:,ii:ii+1])      
        offset_result = seis_forward.vel_to_seis(vel_offset, d.seismogram)[0]
        #offset_file = kgs.dill_load(kgs.temp_dir + 'nondiff')
        finite_difference[ii] = (offset_result.data - base_result.data).flatten()
        #for jj in range(len(offset_file)):
        #    finite_difference_file[ii,jj] = offset_file[jj]-base_file[jj]


    #diff_result = seis_forward.vel_to_seis_diff(d.velocity, offset_vecs)
    #diff_result_file = kgs.dill_load(kgs.temp_dir + 'diff')
    #for ii in range(N_test):
    #    for jj in range(len(offset_file)):
    #        print (kgs.rms(diff_result_file[jj][ii,...] - finite_difference_file[ii,jj]),kgs.rms(finite_difference_file[ii,jj]))
    for ii in range(N_test):
        assert (kgs.rms(diff_result[:,ii] - finite_difference[ii])/kgs.rms(finite_difference[ii])) < 1e-5

            
    

    d.unload()

def test_prior(prior):

    prior.check_constraints()
    prior.λ = 15.

    basis_functions = prior.basis_functions()
    plt.figure()
    plt.imshow(cp.asnumpy(basis_functions), aspect='auto', cmap='bone', interpolation='none')
    plt.colorbar()

    base_vec = cp.array(np.random.default_rng(seed=0).normal(0,1,(prior.N,1)), dtype=kgs.base_type_gpu)
    offset_vec = 1e-6*cp.array(np.random.default_rng(seed=0).normal(0,1,(prior.N,1)), dtype=kgs.base_type_gpu)

    cost,gradient = prior.compute_cost_and_gradient(base_vec, compute_gradient=True)
    cost_offset = prior.compute_cost_and_gradient(base_vec+offset_vec)

    assert kgs.rms(cp.sum(gradient*offset_vec) - (cost_offset-cost))/kgs.rms(cost_offset-cost) < 1e-6

def test_cost(data, prior):

    data.load_to_memory()

    prior.check_constraints()
    data.check_constraints()
    prior.λ = 1e-9

    base_vec = cp.array(np.random.default_rng(seed=0).normal(0,1,(prior.N,1)), dtype=kgs.base_type_gpu)
    offset_vec = 1e-4*cp.array(np.random.default_rng(seed=0).normal(0,1,(prior.N,1)), dtype=kgs.base_type_gpu)
    target = data.seismogram.to_vector()
    basis_functions = prior.basis_functions()

    cost_offset = seis_invert.cost_and_gradient(base_vec+offset_vec, target, prior, basis_functions, compute_gradient=False)
    cost,gradient = seis_invert.cost_and_gradient(base_vec, target, prior, basis_functions, compute_gradient=True)

    assert (kgs.rms(cp.sum(gradient*offset_vec) - (cost_offset-cost))/kgs.rms(cost_offset-cost))<1e-4
    print (kgs.rms(cp.sum(gradient*offset_vec) - (cost_offset-cost)), kgs.rms(cost_offset-cost))

def run_all_tests(test_reference_mode = False):
    importlib.reload(kgs)
    importlib.reload(seis_forward)
    importlib.reload(seis_prior)
    importlib.reload(seis_invert)
    kgs.debugging_mode = 3
    kgs.profiling=False
    seis_forward.reference_mode = False
    data = kgs.load_all_train_data()
    
    test_stuff_on_one_case(data[2059], 1e-4, test_reference_mode=test_reference_mode)
    return
    test_stuff_on_one_case(data[-1001], 1e-4, test_reference_mode=test_reference_mode)

    test_prior(seis_prior.RowTotalVariation())

    test_cost(data[2059], seis_prior.RowTotalVariation())
    
    importlib.reload(kgs)
    importlib.reload(seis_forward)
    importlib.reload(seis_prior)
    importlib.reload(seis_invert)

    print('All tests passed!')
    