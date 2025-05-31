import numpy as np
import cupy as cp
import kaggle_support as kgs
from dataclasses import dataclass, field, fields

def cost_and_gradient(x, target, prior):

    # Prior part
    cost_prior, gradient_prior = 