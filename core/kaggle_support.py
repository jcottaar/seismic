import pandas as pd
import numpy as np
import scipy as sp
import dill # like pickle but more powerful
import itertools
import os
import copy
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import IPython
from dataclasses import dataclass, field, fields
import enum
import typing
import pathlib
import multiprocess
multiprocess.set_start_method('spawn', force=True)
from decorator import decorator
from line_profiler import LineProfiler
import os
import gc
import torch
import concurrent
import glob
import cv2
import h5py
import time
import sklearn
import skimage
import shutil
import subprocess
import inspect
import csv


'''
Determine environment and globals
'''

if os.path.isdir('f:/seismic/'):
    env = 'local'
elif os.path.isdir('/kaggle/working/'):
    env = 'kaggle'
else:
    env = 'vast';
assert not env=='vast'

profiling = False
debugging_mode = 2
verbosity = 1

match env:
    case 'local':
        data_dir = 'f:/seismic/data/'
        temp_dir = 'f:/seismic/temp/'             
        code_dir = 'f:/seismic/code/core/' 
        cache_dir = 'f:/seismic/cache/'
        output_dir = temp_dir
        brendan_model_dir = 'F:/seismic/models/brendan/'
    case 'kaggle':
        data_dir = '/kaggle/input/waveform-inversion/'
        temp_dir = '/temp/'             
        code_dir = '/kaggle/input/my-seismic-library/' 
        cache_dir = '/kaggle/working/cache/'
        output_dir = '/kaggle/working/'
        brendan_model_dir = '/kaggle/input/simple-further-finetuned-bartley-open-models/'
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# How many workers is optimal for parallel pool?
def recommend_n_workers():
    return torch.cuda.device_count()

n_cuda_devices = recommend_n_workers()
process_name = multiprocess.current_process().name
if not multiprocess.current_process().name == "MainProcess":
    pid = int(multiprocess.current_process().name[-1])-1    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(np.mod(pid, n_cuda_devices))
    print('CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"]);

import cupy as cp

base_type = np.float64
base_type_gpu = cp.float64


'''
Helper classes and functions
'''

def list_attrs(obj):
    for name, val in inspect.getmembers(obj):
        if name.startswith("_"):
            continue
        # skip methods, but let descriptors through
        if callable(val) and not isinstance(val, property):
            continue
        print(f"{name} = {val}")

def remove_and_make_dir(path):
    try: shutil.rmtree(path)
    except: pass
    os.makedirs(path)

# Helper class - doesn't allow new properties after construction, and enforces property types. Partially written by ChatGPT.
@dataclass
class BaseClass:
    _is_frozen: bool = field(default=False, init=False, repr=False)

    def check_constraints(self, debugging_mode_offset = 0):
        global debugging_mode
        debugging_mode = debugging_mode+debugging_mode_offset
        try:
            if debugging_mode > 0:
                self._check_types()
                self._check_constraints()
            return
        finally:
            debugging_mode = debugging_mode - debugging_mode_offset

    def _check_constraints(self):
        pass

    def _check_types(self):
        type_hints = typing.get_type_hints(self.__class__)
        for field_info in fields(self):
            field_name = field_info.name
            expected_type = type_hints.get(field_name)
            actual_value = getattr(self, field_name)
            
            if expected_type and not isinstance(actual_value, expected_type) and not actual_value is None:
                raise TypeError(
                    f"Field '{field_name}' expected type {expected_type}, "
                    f"but got value {actual_value} of type {type(actual_value).__name__}.")

    def __post_init__(self):
        # Mark the object as frozen after initialization
        object.__setattr__(self, '_is_frozen', True)

    def __setattr__(self, key, value):
        # If the object is frozen, prevent setting new attributes
        if self._is_frozen and not hasattr(self, key):
            raise AttributeError(f"Cannot add new attribute '{key}' to frozen instance")
        super().__setattr__(key, value)

# Small wrapper for dill loading
def dill_load(filename):
    filehandler = open(filename, 'rb');
    data = dill.load(filehandler)
    filehandler.close()
    return data

# Small wrapper for dill saving
def dill_save(filename, data):
    filehandler = open(filename, 'wb');
    data = dill.dump(data, filehandler)
    filehandler.close()
    return data

def prep_pytorch(seed, deterministic, deterministic_needs_cpu):
    if seed is None:
        seed = np.random.default_rng(seed=None).integers(0,1e6).item()
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.set_num_threads(1)
    cpu = torch.device("cpu")
    if deterministic and deterministic_needs_cpu:
        device = cpu
    else:
        device = torch.device("cuda")
        claim_gpu('pytorch')
    return cpu, device

gpu_claimant = ''
def claim_gpu(new_claimant):
    global gpu_claimant
    old_claimant = gpu_claimant
    gpu_claimant = new_claimant
    if new_claimant == old_claimant or old_claimant == '':
        return
    gc.collect()
    if old_claimant == 'cupy':
        print('Clearing cupy')
        import cupy # can't do earlier or it will select wrong device
        cache = cupy.fft.config.get_plan_cache()
        cache.clear()
        cupy.get_default_memory_pool().free_all_blocks()
    elif old_claimant == 'pytorch':
        print('Clearing pytorch')
        import torch
        torch.cuda.empty_cache()
    else:
        raise Exception('Unrecognized GPU claimant')

@decorator
def profile_each_line(func, *args, **kwargs):
    if not profiling:
        return func(*args, **kwargs)
    profiler = LineProfiler()
    profiled_func = profiler(func)
    try:
        s=profiled_func(*args, **kwargs)
        profiler.print_stats()
        return s
    except:
        profiler.print_stats()
        raise

def profile_print(string):
    if profiling: print(string)

def download_kaggle_dataset(dataset_name, destination, skip_download=False):
    remove_and_make_dir(destination)
    if not skip_download:
        subprocess.run('kaggle datasets download ' + dataset_name + ' --unzip -p ' + destination, shell=True)
    subprocess.run('kaggle datasets metadata -p ' + destination + ' ' + dataset_name, shell=True)

def upload_kaggle_dataset(source):
    if env=='local':
        source=source.replace('/', '\\')
    subprocess.run('kaggle datasets version -p ' + source + ' -m ''Update''', shell=True)

def rms(array):
    return np.sqrt(np.mean(array**2))

'''
Data definition and loading
'''
@dataclass
class Seismogram(BaseClass):
    filename: str = field(init=True, default=None)
    ind: int = field(init=True, default=None)    
    data: cp.ndarray = field(init=True, default=None) # 5x999x70

    def _check_constraints(self):
        if not self.data is None:
            assert(self.data.dtype == base_type_gpu)
            assert(self.data.shape == (5,999,70) or self.data.shape == (5,1000,70))

    def load_to_memory(self, load_last_row=False):
        if self.ind is None:
            self.data = cp.array( np.load(self.filename, mmap_mode='r')[:,:999+load_last_row,:], dtype = base_type_gpu )
        else:
            self.data = cp.array( np.load(self.filename, mmap_mode='r')[self.ind,:,:999+load_last_row,:], dtype = base_type_gpu )

    def to_vector(self):
        vec = self.data.flatten()[:,None]
        return vec

    def from_vector(self, vec):
        assert vec.shape == (5*999*70,1)
        self.data = cp.reshape(vec, (5,999,70))
        if debugging_mode >= 2:
            assert cp.all(self.to_vector()==vec)

    def unload(self):
        self.data = None

@dataclass
class Velocity(BaseClass):
    filename: str = field(init=True, default=None)
    ind: int = field(init=True, default=None)    
    data: object = field(init=True, default=None) # 70x70, cp or np array
    min_vel: object = field(init=True, default=None) 

    def _check_constraints(self):
        if not self.data is None:
            assert(self.data.shape == (70,70))
            #assert(self.data.dtype == base_type_gpu)
            assert(self.min_vel.shape == ())
            #assert(self.min_vel.dtype == base_type_gpu)

    def load_to_memory(self):
        self.data = cp.array( np.load(self.filename, mmap_mode='r')[self.ind,0,:,:], dtype = base_type_gpu )
        self.min_vel = cp.min(self.data)

    def to_vector(self):
        vec = cp.concatenate((self.data.flatten(), cp.reshape(self.min_vel, (1))))[:,None]
        return vec

    def from_vector(self, vec):
        assert vec.shape == (4901,1)
        self.data = cp.reshape(vec[:-1,0], (70,70))
        self.min_vel = vec[-1,0]
        if debugging_mode >= 2:
            assert cp.all(self.to_vector()==vec)
        

    def unload(self):
        self.data = None
    
@dataclass
class Data(BaseClass):
    is_train: bool = field(init=True, default=False)
    family: str = field(init=True, default='')

    seismogram: Seismogram = field(init=True, default_factory=lambda:Seismogram() )
    velocity: Velocity = field(init=True, default=None )    
    velocity_guess: Velocity = field(init=True, default=None )    

    diagnostics: dict = field(init=True, default_factory=dict)

    def _check_constraints(self):
        self.seismogram.check_constraints()
        if not self.velocity is None:
            self.velocity.check_constraints()
        if not self.velocity_guess is None:
            self.velocity_guess.check_constraints()

    def load_to_memory(self):
        self.seismogram.load_to_memory()
        if not self.velocity is None:
            self.velocity.load_to_memory()
        self.check_constraints()

    def unload(self):
        self.seismogram.unload()
        if not self.velocity is None:
            self.velocity.unload()
        self.check_constraints()

    def cache_name(self):
        return os.path.basename(self.seismogram.filename[:-4]+'__'+self.family+'__'+str(self.seismogram.ind))

def load_all_train_data(validation_only = False):
    dirs = glob.glob(data_dir + '/train_samples/*')
    base_data = Data()
    base_data.is_train = True
    base_data.velocity = Velocity()
    data_list = []
    for d in dirs:
        family_ind = max(d.rfind('/'), d.rfind('\\'))
        base_data.family = d[family_ind+1:]
        files = glob.glob(d + '/seis*.npy')
        files.sort()
        if validation_only:
            files = files[0:1]
        for f in files:
            #print(f)
            ind1 = f.rfind('seis')
            ind2 = f.rfind('.')
            descriptor = f[ind1+4:ind2]
            base_data.seismogram.filename = d+'/seis'+descriptor+'.npy'
            base_data.velocity.filename = d+'/vel'+descriptor+'.npy'
            for ind in range(np.load(base_data.seismogram.filename, mmap_mode='r').shape[0]):
                data_list.append(copy.deepcopy(base_data))
                data_list[-1].seismogram.ind = ind
                data_list[-1].velocity.ind = ind
            data_list[-1].check_constraints()

        if os.path.isdir(d+'/data/'):            
            for ii in range(len(glob.glob(d+'/data/*.npy'))):
                if validation_only and not ii==0:
                    continue
                base_data.seismogram.filename = d+'/data/data'+str(ii+1)+'.npy'
                base_data.velocity.filename = d+'/model/model'+str(ii+1)+'.npy'
                #print(base_data.seismogram.filename)
                for ind in range(np.load(base_data.seismogram.filename, mmap_mode='r').shape[0]):
                    data_list.append(copy.deepcopy(base_data))
                    data_list[-1].seismogram.ind = ind
                    data_list[-1].velocity.ind = ind
                data_list[-1].check_constraints()

    return data_list

def load_all_test_data():
    files = glob.glob(data_dir + '/test/*')
    data_list = []
    base_data = Data()
    base_data.is_train = False
    base_data.family = 'test'
    base_data.seismogram.ind = None
    for f in files:
        data_list.append(copy.deepcopy(base_data))
        data_list[-1].seismogram.filename = f        
        data_list[-1].check_constraints()

    return data_list
        

    
'''
General model definition
'''
# Function is used below, I ran into issues with multiprocessing if it was not a top-level function
model_parallel = None
def infer_internal_single_parallel(data):    
    try:
        global model_parallel
        if model_parallel is None:
            model_parallel= dill_load(temp_dir+'parallel.pickle')
        data.seismogram.load_to_memory()
        return_data = model_parallel._infer_single(data)
        return_data.seismogram.unload()
        return return_data
    except Exception as err:
        import traceback
        print(traceback.format_exc())     
        raise


@dataclass
class Model(BaseClass):
    # Loads one or more cryoET measuerements
    state: int = field(init=False, default=0) # 0: untrained, 1: trained    
    run_in_parallel: bool = field(init=False, default=False) 
    seed: object = field(init=True, default=None)  
    cache_name: str = field(init=True, default=None)

    write_cache: bool = field(init=True, default=False)
    read_cache: bool = field(init=True, default=False)

    def _check_constraints(self):
        assert(self.state>=0 and self.state<=1)

    def train(self, train_data, validation_data):
        if self.state>=1:
            return
        if self.seed is None:
            self.seed = np.random.default_rng(seed=None).integers(0,1e6).item()
        train_data = copy.deepcopy(train_data)
        validation_data = copy.deepcopy(validation_data)
        for d in train_data:
            d.unload()
        for d in validation_data:
            d.unload()
        self._train(train_data, validation_data)
        for d in train_data:
            d.unload()
        for d in validation_data:
            d.unload()
        self.state = 1
        self.check_constraints()        

    def _train(self,train_data, validation_data):
        pass
        # No training needed if not overridden

    @profile_each_line
    def infer(self, test_data):
        assert self.state == 1
        test_data = copy.deepcopy(test_data)

        if self.read_cache:
            this_cache_dir = cache_dir+self.cache_name+'/'
            files = set([os.path.basename(x) for x in glob.glob(this_cache_dir+'/*')])
            cached = []
            test_data_cached = []
            tt = copy.deepcopy(test_data)
            test_data = []
            for d in tt:
                if d.cache_name() in files:
                    cached.append(True)
                    test_data_cached.append(d)
                    test_data_cached[-1].velocity_guess = dill_load(this_cache_dir+d.cache_name())
                else:
                    cached.append(False)
                    test_data.append(d)
       
        for t in test_data:
            t.unload()
        if len(test_data)>0:
            test_data_inferred = self._infer(test_data)
        else:
            test_data_inferred = []

        if self.read_cache:
            b_it = iter(test_data_cached)
            c_it = iter(test_data_inferred)        
            test_data = [
                next(b_it) if c else next(c_it)
                for c in cached
            ] 
        else:
            test_data = test_data_inferred

        for t in test_data:
            t.seismogram.unload()
            t.check_constraints()

        if self.write_cache:
            this_cache_dir = cache_dir+self.cache_name+'/'
            os.makedirs(this_cache_dir,exist_ok=True)
            for d in test_data:
                dill_save(this_cache_dir+d.cache_name(), d.velocity_guess)
                
        return test_data

    def _infer(self, test_data):
        # Subclass must implement this OR _infer_single
        if self.run_in_parallel:
            for t in test_data:
                t.unload()
            claim_gpu('')
            with multiprocess.Pool(recommend_n_workers()) as p:
                dill_save(temp_dir+'parallel.pickle', self)
                result = p.starmap(infer_internal_single_parallel, zip(test_data))            
        else:
            result = []
            for xx in test_data:     
                x = copy.deepcopy(xx)  
                if x.seismogram.data is None:
                    x.seismogram.load_to_memory()
                x = self._infer_single(x)
                x.seismogram.unload()       
                if self.write_cache: # will be done later too, but in case we error out later...
                    this_cache_dir = cache_dir+self.cache_name+'/'
                    os.makedirs(this_cache_dir,exist_ok=True)
                    dill_save(this_cache_dir+x.cache_name(), x.velocity_guess)
                result.append(x)
        result = self._post_process(result)
        return result

    def _post_process(self, result):
        return result

@dataclass
class ChainedModel(Model):
    models: list = field(init=True, default_factory=list)

    def _train(self,train_data, validation_data):
        for m in self.models:
            m.train(train_data, validation_data)

    def _infer(self, data):
        for m in self.models:
            data = m.infer(data)
        return data

def score_metric(data, show_diagnostics=True):
    res_all = []
    res_per_family = dict()
    for d in data:
        d.velocity.load_to_memory()
        this_error = np.mean(np.abs(cp.asnumpy(d.velocity.data) - d.velocity_guess.data))
        d.velocity.unload()
        res_all.append(this_error)
        if not d.family in res_per_family:
            res_per_family[d.family] = []
        res_per_family[d.family].append(this_error)

    score = np.mean(res_all)
    score_per_family = dict()
    score_per_family['family'] = res_per_family.keys()
    score_per_family['score'] = [np.mean(x) for x in res_per_family.values()]
    score_per_family = pd.DataFrame(score_per_family)

    if show_diagnostics:
        print(score_per_family)
        print('Combined: ', score)

    return score,score_per_family,res_all

def write_submission_file(data, output_file = output_dir+'submission.csv'):
    res = dict()
    res['oid_ypos'] = []
    x_vals = np.arange(1,70,2)
    x_vals_names = [ 'x_'+str(x) for x in x_vals ]
    for xn in x_vals_names:
        res[xn] = []
    for ii,d in enumerate(data):
        if ii%100==0:print(ii)
        name = os.path.basename(d.seismogram.filename[:-4])+'_y_'
        data = np.round(d.velocity_guess.data).astype(int)
        for y in np.arange(70):
            res['oid_ypos'].append(name+str(y))
            for x,xn in zip(x_vals, x_vals_names):
                res[xn].append(data[y,x])
    print('x')
    df = pd.DataFrame(res)
    print('xx')
    df.to_csv(output_file, index=False)
            
def write_submission_file(data, output_file = output_dir+'submission.csv'):
    # precompute x‐positions and header
    x_vals = np.arange(1, 70, 2)
    x_names = [f"x_{x}" for x in x_vals]
    header = ["oid_ypos"] + x_names

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ii, d in enumerate(data):

            # your string prefix
            base = os.path.basename(d.seismogram.filename)[:-4]
            name_prefix = f"{base}_y_"

            # grab and round your 70×70 numpy array
            arr = d.velocity_guess.data.astype(np.float32)

            # slice out only the 35 columns you care about
            sub = arr[:, x_vals]  # shape = (70, 35)

            # stream each of the 70 rows
            for y in range(sub.shape[0]):
                writer.writerow([f"{name_prefix}{y}"] + sub[y].tolist())