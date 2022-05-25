#%%
import numpy as np
from qutip import *
# %%
def partial_trace(Op, dims, to_trace):
    N = len(dims)
    new_dims = [dims[i] for i in range(N) if i not in to_trace]
    new_op = np.zeros((np.prod(new_dims),np.prod(new_dims)), dtype=np.complex64)
    for i in range(new_op.shape[0]):
        for j in range(new_op.shape[1]):
            for k in to_trace:
                for m in range(dims[k]):
                    new_op[i,j] += Op[i+k*m,j+k*m]

    return new_op
    

# %%
#state= tensor(1*basis(3,0) + 2*basis(3,1) + 3*basis(3,2), 1*basis(4,0) + 2*basis(4,1) + 3*basis(4,2) + 4*basis(4,3))
state = tensor(1*basis(2,0) + 2*basis(2,1) , 3*basis(2,0) + 4*basis(2,1))
M = state * state.dag()
print(M)
print(ptrace(M,[0]))
# %%
Op = np.array(M)
dims = [2,2]
to_trace = [1]
new_op = partial_trace(Op, dims, to_trace)
print(new_op)
# %%
