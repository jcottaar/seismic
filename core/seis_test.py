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

    # Test prep_run_diff
    offset_vecs = 1e-3*cp.array(np.random.default_rng(seed=0).normal(0,1,(2,len(d.velocity.to_vector()))), dtype=kgs.base_type_gpu)
    base_result = seis_forward.prep_run(d.velocity)
    
    finite_difference = dict()
    for ii in range(2):
        vel_offset = copy.deepcopy(d.velocity)    
        vel_offset.from_vector(vel_offset.to_vector() + offset_vecs[ii,:])      
        offset_result = seis_forward.prep_run(vel_offset)
        for jj in range(len(offset_result)):
            finite_difference[ii,jj] = offset_result[jj]-base_result[jj]

    diff_result = seis_forward.prep_run_diff(d.velocity, cp.reshape(offset_vecs[:,:-1],(2,70,70)), offset_vecs[:,-1])
    for ii in range(2):
        for jj in range(len(offset_result)):
            assert (kgs.rms(diff_result[jj][ii,...] - finite_difference[ii,jj])/kgs.rms(finite_difference[ii,jj])) < 1e-6
            
                     
    # Test matching to host's forward model
    seis_pred = seis_forward.vel_to_seis(d.velocity,d.seismogram)
    mismatch = kgs.rms(seis_pred.data - d.seismogram.data)
    print(mismatch)
    assert mismatch < expected_match
    if test_reference_mode:
        seis_forward.reference_mode = True
        seis_pred_ref = seis_forward.vel_to_seis(d.velocity,d.seismogram)
        print(kgs.rms(seis_pred.data - seis_pred_ref.data), kgs.rms(seis_pred_ref.data - d.seismogram.data))
        seis_forward.reference_mode = False
        

    

    d.unload()

def run_all_tests(test_reference_mode = False):
    importlib.reload(kgs)
    importlib.reload(seis_forward)
    kgs.debugging_mode = 2
    seis_forward.reference_mode = False
    data = kgs.load_all_train_data()
    test_stuff_on_one_case(data[2059], 1e-4, test_reference_mode=test_reference_mode)
    importlib.reload(kgs)
    importlib.reload(seis_forward)

    print('All tests passed!')
    