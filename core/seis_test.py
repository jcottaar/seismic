# Original source: XXX
# Python translation adapted from: https://www.kaggle.com/code/jaewook704/waveform-inversion-vel-to-seis by Jae-Wook Kim

import numpy as np
import cupy as cp
import kaggle_support as kgs
import seis_forward
import copy
import importlib

def test_stuff_on_one_case(d, expected_match):

    d.load_to_memory()

    # Check vector transforms
    d.velocity.from_vector(d.velocity.to_vector())
    d.seismogram.from_vector(d.seismogram.to_vector())

    # Test matching to host's forward model
    seis_pred = seis_forward.vel_to_seis(d.velocity,d.seismogram)
    mismatch = kgs.rms(seis_pred.data - d.seismogram.data)
    print(mismatch)
    assert mismatch < expected_match
    seis_forward.reference_mode = True
    #seis_pred_ref = seis_forward.vel_to_seis(d.velocity,d.seismogram)
    #print(kgs.rms(seis_pred.data - seis_pred_ref.data))

    

    d.unload()

def run_all_tests():
    importlib.reload(kgs)
    importlib.reload(seis_forward)
    kgs.debugging_mode = 2
    seis_forward.reference_mode = False
    data = kgs.load_all_train_data()
    test_stuff_on_one_case(data[2059], 1e-4)

    print('All tests passed!')
    