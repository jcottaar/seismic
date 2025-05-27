# Original source: XXX
# Python translation adapted from: https://www.kaggle.com/code/jaewook704/waveform-inversion-vel-to-seis by Jae-Wook Kim

import numpy as np
import cupy as cp
import kaggle_support as kgs
import seis_forward
import copy
import importlib

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
        seis_forward.reference_mode = True
        seis_pred_ref = seis_forward.vel_to_seis(d.velocity,d.seismogram)[0]
        assert (kgs.rms(seis_pred.data - seis_pred_ref.data)/ kgs.rms(seis_pred_ref.data - d.seismogram.data))<1e-5
        seis_forward.reference_mode = False
        

    # Test prep_run_diff
    N_test = 2
    offset_vecs = 1e-3*cp.array(np.random.default_rng(seed=0).normal(0,1,(N_test,len(d.velocity.to_vector()))), dtype=kgs.base_type_gpu)
    base_result = seis_forward.prep_run(d.velocity,3)
    
    finite_difference = dict()
    for ii in range(N_test):
        vel_offset = copy.deepcopy(d.velocity)    
        vel_offset.from_vector(vel_offset.to_vector() + offset_vecs[ii,:])      
        offset_result = seis_forward.prep_run(vel_offset,3)
        for jj in range(len(offset_result)):
            finite_difference[ii,jj] = offset_result[jj]-base_result[jj]

    diff_result = seis_forward.prep_run_diff(d.velocity, cp.reshape(offset_vecs[:,:-1],(N_test,70,70)), offset_vecs[:,-1],3)
    for ii in range(N_test):
        for jj in range(len(offset_result)):
            assert (kgs.rms(diff_result[jj][ii,...] - finite_difference[ii,jj])/kgs.rms(finite_difference[ii,jj])) < 1e-6

    # Test vel_to_seis_diff
    N_test = 2
    offset_vecs = 1e-4*cp.array(np.random.default_rng(seed=0).normal(0,1,(N_test,len(d.velocity.to_vector()))), dtype=kgs.base_type_gpu)
    base_result, diff_result = seis_forward.vel_to_seis(d.velocity, d.seismogram, offset_vecs)
    #base_file = kgs.dill_load(kgs.temp_dir + 'nondiff')
    
    finite_difference = dict()
    #finite_difference_file = dict()
    for ii in range(N_test):
        vel_offset = copy.deepcopy(d.velocity)    
        vel_offset.from_vector(vel_offset.to_vector() + offset_vecs[ii,:])      
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
        assert (kgs.rms(diff_result[ii,:] - finite_difference[ii])/kgs.rms(finite_difference[ii])) < 1e-5

            
    

    d.unload()

def run_all_tests(test_reference_mode = False):
    importlib.reload(kgs)
    importlib.reload(seis_forward)
    kgs.debugging_mode = 2
    kgs.profiling=False
    seis_forward.reference_mode = False
    data = kgs.load_all_train_data()
    test_stuff_on_one_case(data[2059], 1e-4, test_reference_mode=test_reference_mode)
    test_stuff_on_one_case(data[-1001], 1e-4, test_reference_mode=test_reference_mode)
    importlib.reload(kgs)
    importlib.reload(seis_forward)

    print('All tests passed!')
    