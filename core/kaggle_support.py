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


'''
Determine environment and globals
'''

if os.path.isdir('f:/seismic/'):
    env = 'local'
elif os.path.isdir('/kaggle/working/'):
    env = 'kaggle'
else:
    env = 'vast';
assert env=='local'

profiling = False
debugging_mode = 2
verbosity = 1

match env:
    case 'local':
        data_dir = 'f:/seismic/data/'
        temp_dir = 'f:/seismic/temp/'             
        code_dir = 'f:/seismic/code/core/' 
os.makedirs(temp_dir, exist_ok=True)

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
            assert(self.data.shape == (5,999,70))

    def load_to_memory(self):
        self.data = cp.array( np.load(self.filename, mmap_mode='r')[self.ind,:,:999,:], dtype = base_type_gpu )

    def unload(self):
        self.data = None

@dataclass
class Velocity(BaseClass):
    filename: str = field(init=True, default=None)
    ind: int = field(init=True, default=None)    
    data: cp.ndarray = field(init=True, default=None) # 70x70
    min_vel: cp.ndarray = field(init=True, default=None) 

    def _check_constraints(self):
        if not self.data is None:
            assert(self.data.shape == (70,70))
            assert(self.data.dtype == base_type_gpu)
            assert(self.min_vel.shape == ())
            assert(self.min_vel.dtype == base_type_gpu)

    def load_to_memory(self):
        self.data = cp.array( np.load(self.filename, mmap_mode='r')[self.ind,0,:,:], dtype = base_type_gpu )
        self.min_vel = cp.min(self.data)

    def to_vector(self):
        vec = cp.concatenate((self.data.flatten(), cp.reshape(self.min_vel, (1))))
        return vec

    def from_vector(self, vec):
        self.data = cp.reshape(vec[:-1], (70,70))
        self.min_vel = vec[-1]
        if debugging_mode >= 2:
            assert cp.all(self.to_vector()==vec)
        

    def unload(self):
        self.data = None
    
@dataclass
class Data(BaseClass):
    is_train: bool = field(init=True, default=False)
    family: str = field(init=True, default='')

    seismogram: Seismogram = field(init=True, default=Seismogram() )
    velocity: Velocity = field(init=True, default=None )    

    def _check_constraints(self):
        self.seismogram.check_constraints()
        if not self.velocity is None:
            self.velocity.check_constraints()

    def load_to_memory(self):
        self.seismogram.load_to_memory()
        self.velocity.load_to_memory()
        self.check_constraints()

    def unload(self):
        self.seismogram.unload()
        self.velocity.unload()
        self.check_constraints()

def load_all_train_data():
    dirs = glob.glob(data_dir + '/train_samples/*')
    base_data = Data()
    base_data.is_train = True
    base_data.velocity = Velocity()
    data_list = []
    for d in dirs:
        family_ind = max(d.rfind('/'), d.rfind('\\'))
        base_data.family = d[family_ind+1:]
        files = glob.glob(d + '/seis*.npy')
        for f in files:
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
                base_data.seismogram.filename = d+'/data/data'+str(ii+1)+'.npy'
                base_data.velocity.filename = d+'/model/model'+str(ii+1)+'.npy'
                for ind in range(np.load(base_data.seismogram.filename, mmap_mode='r').shape[0]):
                    data_list.append(copy.deepcopy(base_data))
                    data_list[-1].seismogram.ind = ind
                    data_list[-1].velocity.ind = ind
                data_list[-1].check_constraints()

    return data_list

# def load_one_measurement(name, is_train, include_train_labels):
#     result = DataKaggle()
#     result.name = name
#     result.is_train = is_train
#     if include_train_labels:
#         assert is_train
#         this_labels = copy.deepcopy(all_train_labels[all_train_labels['tomo_id']==name]).reset_index()
#         result.labels = this_labels[['z', 'y', 'x']]
#         if result.labels['z'][0]==-1:
#             assert result.labels['y'][0]==-1
#             assert result.labels['x'][0]==-1
#             assert len(result.labels)==1
#             result.labels = result.labels[0:0]
#         result.voxel_spacing = this_labels[0:1][['Voxel spacing']].to_numpy()[0,0]
#         result.data_shape = (this_labels[0:1][['Array shape (axis 0)']].to_numpy()[0,0], this_labels[0:1][['Array shape (axis 1)']].to_numpy()[0,0], this_labels[0:1][['Array shape (axis 2)']].to_numpy()[0,0])
#         result.negative_labels = negative_labels[negative_labels['name']==name].reset_index()
#     result.check_constraints()    
#     return result

# def load_one_measurement_extra(name, include_train_labels):
#     result = DataExtra()
#     result.name = name
#     result.is_train = True        
#     if include_train_labels:
#         data_pickle = dill_load(data_dir + '/extra/' + name + '/info.pickle')
#         result.voxel_spacing = data_pickle['voxel_spacing']
#         result.labels = data_pickle['labels']
#         result.data_shape = data_pickle['orig_size']
#     result.check_constraints()    
#     return result

# def load_all_train_data():    
#     #if env=='vast':
#     #    directories = glob.glob(h5py_cache_dir + '*.h5')
#     #else:
#     directories = glob.glob(data_dir + 'train/tomo*')
#     directories.sort()
#     result = []
#     for d in directories:
#         name = d[max(d.rfind('\\'), d.rfind('/'))+1:]
#         # if env=='vast':
#         #     name = name[:-3]
#         if not name in['tomo_2b3cdf', 'tomo_62eea8', 'tomo_c84b8e', 'tomo_e6f7f7']: # mislabeled
#             result.append(load_one_measurement(name, True, True))
#     return result

# def load_all_test_data():
#     directories = glob.glob(data_dir + 'test/tomo*')
#     directories.sort()
#     result = []
#     for d in directories:
#         name = d[max(d.rfind('\\'), d.rfind('/'))+1:]
#         result.append(load_one_measurement(name, False, False))
#     return result

# def load_all_extra_data():
#     files = glob.glob(data_dir + 'extra/*')
#     files.sort()
#     result = []
#     for f in files:
#         name = f[max(f.rfind('\\'), f.rfind('/'))+1:]
#         result.append(load_one_measurement_extra(name, True))
#     return result
    
# '''
# General model definition
# '''
# # Function is used below, I ran into issues with multiprocessing if it was not a top-level function
# model_parallel = None
# def infer_internal_single_parallel(data):    
#     try:
#         global model_parallel
#         if model_parallel is None:
#             model_parallel= dill_load(temp_dir+'parallel.pickle')
#         return_data = model_parallel._infer_single(data)
#         return_data.unload()
#         return return_data
#     except Exception as err:
#         import traceback
#         print(traceback.format_exc())     
#         raise

# def train_parallel(model, train_data, validation_data):
#     old_run_in_parallel = model.run_in_parallel
#     model.run_in_parallel = False
#     model.train(train_data, validation_data)
#     model.run_in_parallel = old_run_in_parallel
#     return model

# @dataclass
# class DataSelector(BaseClass):
#     datasets: list = field(init=False, default_factory = lambda:['tom'])#['tom', 'ycw', 'aba', 'mba'])
#     include_multi_motor: bool = field(init=False, default=True)

#     def select(self,data):
#         if not self.include_multi_motor:
#             data_out = []
#             for d in data:
#                 if len(d.labels)<=1:
#                     data_out.append(d)
#             data = data_out

#         data_out = []
#         for d in data:
#             if d.name[:3] in self.datasets:
#                 data_out.append(d)
#         data = data_out
#         return data

# @dataclass
# class Model(BaseClass):
#     # Loads one or more cryoET measuerements
#     state: int = field(init=False, default=0) # 0: untrained, 1: trained
#     quiet: bool = field(init=False, default=True)
#     run_in_parallel: bool = field(init=False, default=True) 
#     seed: object = field(init=True, default=None)

#     train_data_selector: object = field(init=True, default_factory = DataSelector)
#     preprocessor: object = field(init=True, default = None)
#     ratio_of_motors_allowed: float = field(init=True, default=0.45)

#     def __post_init__(self):
#         super().__post_init__()
#         import flg_preprocess
#         self.preprocessor = flg_preprocess.Preprocessor2()        

#     def _check_constraints(self):
#         assert(self.state>=0 and self.state<=1)

#     def train_subprocess(self, train_data, validation_data):
#         # Note: unlike below must capture result!
#         claim_gpu('')
#         with multiprocess.Pool(1) as p:
#             trained_model = p.starmap(train_parallel, zip([self], [train_data], [validation_data]))[0]
#         trained_model.check_constraints()
#         return trained_model

#     def train(self, train_data, validation_data):
#         if self.state>1:
#             return
#         if self.seed is None:
#             self.seed = np.random.default_rng(seed=None).integers(0,1e6).item()
#         train_data = copy.deepcopy(train_data)
#         validation_data = copy.deepcopy(validation_data)
#         train_data = self.train_data_selector.select(train_data)
#         validation_data = self.train_data_selector.select(validation_data)
#         for d in train_data:
#             d.unload()
#         for d in validation_data:
#             d.unload()
#         self._train(train_data, validation_data)
#         for d in train_data:
#             d.unload()
#         for d in validation_data:
#             d.unload()
#         self.state = 1
#         self.check_constraints()        

#     def _train_real(self, real_data, return_inferred_labels, test_data):
#         pass

#     def infer(self, test_data):
#         assert self.state == 1
#         test_data = copy.deepcopy(test_data)
#         for t in test_data:
#             t.labels  = pd.DataFrame()
#             t.unload()
#         test_data = self._infer(test_data)

#         all_vals = []
#         for d in test_data:
#             if len(d.labels)==0:
#                 all_vals.append(-np.inf)
#             else:
#                 all_vals.append(d.labels['value'][0])
#         inds = np.argsort(all_vals)
#         if self.ratio_of_motors_allowed<1:
#             for ind in inds[:np.round(len(inds)*(1-self.ratio_of_motors_allowed)).astype(int)]:
#                 test_data[ind].labels = test_data[ind].labels[0:0]
        
#         for t in test_data:
#             t.check_constraints()
#         return test_data

#     def _infer(self, test_data):
#         # Subclass must implement this OR _infer_single
#         if self.run_in_parallel:
#             claim_gpu('')
#             with multiprocess.Pool(recommend_n_workers()) as p:
#                 dill_save(temp_dir+'parallel.pickle', self)
#                 result = p.starmap(infer_internal_single_parallel, zip(test_data))            
#         else:
#             result = []
#             for xx in test_data:     
#                 t = time.time()
#                 x = copy.deepcopy(xx)                   
#                 x = self._infer_single(x)
#                 x.unload()
#                 result.append(x)
#                 profile_print(x.name + ' total infer time: ' + str(time.time()-t))
#         result = self._post_process(result)
#         return result

#     def _post_process(self, result):
#         return result

# def mark_tf_pn(data, reference_data, mark_false_negative=False):
#     assert not mark_false_negative # todo
#     for d,r in zip(data,reference_data):
#         assert d.name==r.name
#         #d.labels_unfiltered['tf_pn'] = np.nan
#         for row_d in range(len(d.labels_unfiltered)):#d.labels_unfiltered.iterrows():
#             #print('row_d: ', row_d)
#             is_true_positive = False
#             for row_r in range(len(r.labels)):
#                 coordinate_cols = ['z', 'y', 'x']
#                 loc_d = d.labels_unfiltered[coordinate_cols][row_d:row_d+1].values
#                 loc_r = r.labels[coordinate_cols][row_r:row_r+1].values
#                 #print(loc_d)
#                 #print(loc_r)
#                 distance = np.linalg.norm(loc_d - loc_r)*r.voxel_spacing
#                 #print(distance)
#                 if distance<1000:
#                     is_true_positive = True
#                     break
#             if is_true_positive:
#                 d.labels_unfiltered.at[ d.labels_unfiltered.index[row_d],'tf_pn'] = 0
#             else:
#                 d.labels_unfiltered.at[ d.labels_unfiltered.index[row_d],'tf_pn'] = 1


# def create_submission_dataframe(submission_data, reference_data = load_all_test_data(), include_voxel_spacing = False):

#      #submission = pd.read_csv(data_dir + '/sample_submission.csv')
#     #print(submission)
#     #submission = submission[0:0]
#     #submission = submission.set_index("id")
    
#     rows = []  # Collect rows as a list of lists or tuples
#     #ind = 0
    
#     for dat in submission_data:
#         if len(dat.labels)==0:
#             pass
#             if include_voxel_spacing:
#                 rows.append([dat.name, -1,-1,-1,10,0])
#             else:
#                 rows.append([dat.name, -1,-1,-1])
#         else:
#             assert(len(dat.labels)==1)
#             lab = copy.deepcopy(dat.labels).reset_index()
#             if include_voxel_spacing:
#                 rows.append([dat.name, lab['z'][0], lab['y'][0], lab['x'][0], dat.voxel_spacing, 1])
#             else:
#                 rows.append([dat.name, lab['z'][0], lab['y'][0], lab['x'][0]])

#     all_names = [d.name for d in reference_data]
#     seen_names = [r[0] for r in rows]
#     assert np.all([(name in all_names) for name in seen_names])
#     for name in all_names:
#         if not name in seen_names:
#             if include_voxel_spacing:
#                 rows.append([name, -1,-1,-1,10,0])
#             else:
#                 rows.append([name, -1,-1,-1])
    
#     # Create a new DataFrame from collected rows
#     if include_voxel_spacing:
#         rows_df = pd.DataFrame(rows, columns=["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2", "Voxel spacing", "Has motor"])
#     else:
#         rows_df = pd.DataFrame(rows, columns=["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"])
#     #rows_df = rows_df.set_index("id")

#     return rows_df

# def write_submission_file(submission_data):   

#     rows_df = create_submission_dataframe(submission_data)
#     print(rows_df)
#     rows_df.to_csv(output_dir + 'submission.csv', index=False)


# def score_competition_metric(data, reference_data):
#     def distance_metric(
#         solution: pd.DataFrame,
#         submission: pd.DataFrame,
#         thresh_ratio: float,
#         min_radius: float,
#     ):
#         coordinate_cols = ['Motor axis 0', 'Motor axis 1', 'Motor axis 2']
#         label_tensor = solution[coordinate_cols].values.reshape(len(solution), -1, len(coordinate_cols))
#         predicted_tensor = submission[coordinate_cols].values.reshape(len(submission), -1, len(coordinate_cols))
#         # Find the minimum euclidean distances between the true and predicted points
#         solution['distance'] = np.linalg.norm(label_tensor - predicted_tensor, axis=2).min(axis=1)
#         # Convert thresholds from angstroms to voxels
#         solution['thresholds'] = solution['Voxel spacing'].apply(lambda x: (min_radius * thresh_ratio) / x)
#         solution['predictions'] = submission['Has motor'].values
#         solution.loc[(solution['distance'] > solution['thresholds']) & (solution['Has motor'] == 1) & (submission['Has motor'] == 1), 'predictions'] = 0
#         return solution['predictions'].values
        
#     def score(solution: pd.DataFrame, submission: pd.DataFrame, min_radius: float, beta: float) -> float:
#         """
#         Parameters:
#         solution (pd.DataFrame): DataFrame containing ground truth motor positions.
#         submission (pd.DataFrame): DataFrame containing predicted motor positions.
    
#         Returns:
#         float: FBeta score.
    
#         Example
#         --------
#         >>> solution = pd.DataFrame({
#         ...     'tomo_id': [0, 1, 2, 3],
#         ...     'Motor axis 0': [-1, 250, 100, 200],
#         ...     'Motor axis 1': [-1, 250, 100, 200],
#         ...     'Motor axis 2': [-1, 250, 100, 200],
#         ...     'Voxel spacing': [10, 10, 10, 10],
#         ...     'Has motor': [0, 1, 1, 1]
#         ... })
#         >>> submission = pd.DataFrame({
#         ...     'tomo_id': [0, 1, 2, 3],
#         ...     'Motor axis 0': [100, 251, 600, -1],
#         ...     'Motor axis 1': [100, 251, 600, -1],
#         ...     'Motor axis 2': [100, 251, 600, -1]
#         ... })
#         >>> score(solution, submission, 1000, 2)
#         0.3571428571428571
#         """
    
#         solution = solution.sort_values('tomo_id').reset_index(drop=True)
#         submission = submission.sort_values('tomo_id').reset_index(drop=True)
    
#         filename_equiv_array = solution['tomo_id'].eq(submission['tomo_id'], fill_value=0).values
    
#         if np.sum(filename_equiv_array) != len(solution['tomo_id']):
#             raise ValueError('Submitted tomo_id values do not match the sample_submission file')
    
#         submission['Has motor'] = 1
#         # If any columns are missing an axis, it's marked with no motor
#         select = (submission[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']] == -1).any(axis='columns')
#         submission.loc[select, 'Has motor'] = 0
    
#         cols = ['Has motor', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']
#         assert all(col in submission.columns for col in cols)
    
#         # Calculate a label of 0 or 1 using the 'has motor', and 'motor axis' values
#         predictions = distance_metric(
#             solution,
#             submission,
#             thresh_ratio=1.0,
#             min_radius=min_radius,
#         )
    
#         return sklearn.metrics.precision_score(solution['Has motor'].values, predictions,zero_division=1.),  sklearn.metrics.recall_score(solution['Has motor'].values, predictions), sklearn.metrics.fbeta_score(solution['Has motor'].values, predictions, beta=beta)

#     row_df_sub = create_submission_dataframe(data, reference_data = reference_data)
#     row_df_ref = create_submission_dataframe(reference_data, reference_data = reference_data, include_voxel_spacing=True)
#     return score(row_df_ref, row_df_sub, 1000, 2)