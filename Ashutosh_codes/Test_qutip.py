#%%
import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import math
# %%
N = 2
freq = 5e9

a = destroy(N)
H = freq*a.dag()*a

