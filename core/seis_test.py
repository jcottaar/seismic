# Original source: XXX
# Python translation adapted from: https://www.kaggle.com/code/jaewook704/waveform-inversion-vel-to-seis by Jae-Wook Kim

import numpy as np
import cupy as cp
import kaggle_support as kgs
import seis_forward
import seis_forward2
import seis_prior
import seis_invert
import seis_model
import copy
import importlib
import matplotlib.pyplot as plt

def test_stuff_on_one_case(d, expected_match, test_reference_mode=False):

    d.load_to_memory()

    # Check vector transforms
    d.velocity.from_vector(d.velocity.to_vector())
    d.seismogram.from_vector(d.seismogram.to_vector())


    # Check vel_to_seis
    offset_vec1 = 1e-4*cp.array(np.random.default_rng(seed=0).normal(0,1,(len(d.velocity.to_vector()),1)), dtype=kgs.base_type_gpu)
    offset_vec2 = d.seismogram.to_vector()+cp.array(np.random.default_rng(seed=0).normal(0,1,(len(d.seismogram.to_vector()),1)), dtype=kgs.base_type_gpu)
    result, result_diff, result_adjoint = seis_forward2.vel_to_seis(d.velocity.to_vector(), offset_vec1, offset_vec2, adjoint_on_residual=True)

    mismatch = kgs.rms(result - d.seismogram.to_vector())
    print('Mismatch to host''s:', mismatch)
    assert mismatch < expected_match    

    res1 = cp.sum((result-offset_vec2)*result_diff)
    res2 = cp.sum(result_adjoint*offset_vec1)
    print('adjoint', cp.abs(res1-res2), res1)
    if kgs.base_type_gpu == cp.float64:
        assert cp.abs(res1-res2)/cp.abs(res1)<1e-10
    #base_file = kgs.dill_load(kgs.temp_dir + 'nondiff')

    result_offset = seis_forward2.vel_to_seis(d.velocity.to_vector()+offset_vec1, None,None)[0]
    print('diff', kgs.rms(result_offset-result-result_diff), kgs.rms(result_offset-result))
    if kgs.base_type_gpu == cp.float64:
        assert kgs.rms(result_offset-result-result_diff)/kgs.rms(result_offset-result)<1e-5

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

    print('prior test', kgs.rms(cp.sum(gradient*offset_vec) - (cost_offset-cost)),kgs.rms(cost_offset-cost))
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

    cost_offset,_,_ = seis_invert.cost_and_gradient(base_vec+offset_vec, target, prior, basis_functions, compute_gradient=False)
    cost,gradient,_,_ = seis_invert.cost_and_gradient(base_vec, target, prior, basis_functions, compute_gradient=True)

    assert (kgs.rms(cp.sum(gradient*offset_vec) - (cost_offset-cost))/kgs.rms(cost_offset-cost))<1e-4
    print (kgs.rms(cp.sum(gradient*offset_vec) - (cost_offset-cost)), kgs.rms(cost_offset-cost))

def test_to_reference(d, write_reference=False):
    print(d.family)
    seis_model.test_mode = True
    model = seis_model.default_model()
    kgs.disable_caching = True
    result = model.infer([d])[0]
    if write_reference:
        assert kgs.env=='local'
        kgs.dill_save(kgs.code_dir + '/' + d.family + '_ref.pickle', result)
    ref = kgs.dill_load(kgs.code_dir + '/' + d.family + '_ref.pickle')
    if kgs.env=='local':
        assert str(result.velocity_guess.data) == str(ref.velocity_guess.data)
    else:
        assert kgs.rms(result.velocity_guess.data - ref.velocity_guess.data)<2
    kgs.disable_caching = False
    seis_model.test_mode = False
    

def run_all_tests(test_reference_mode = False, write_reference=False):
    #kgs.debugging_mode = 3
    #kgs.profiling=False
    #seis_forward.reference_mode = False
    data = kgs.load_all_train_data()

    for d in data[::-1000]:
        test_to_reference(d,write_reference=write_reference)

    test_prior(seis_prior.TotalVariation())
    test_prior(seis_prior.SquaredExponential())
    test_prior(seis_prior.RowTotalVariation())

    test_stuff_on_one_case(data[2059], 1e-4, test_reference_mode=test_reference_mode)
    test_stuff_on_one_case(data[-1001], 1e-4, test_reference_mode=test_reference_mode)

    
    
    
    

    test_cost(data[2059], seis_prior.RowTotalVariation())

    print('All tests passed!')
    