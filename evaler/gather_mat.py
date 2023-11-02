import sys, glob
import numpy as np
from scipy.io import loadmat, savemat
experiment = sys.argv[1]
opath='{}/checkpoints/model_s2_0050.mat'.format(experiment)
npzs =  glob.glob('{}/checkpoints/model_s2_0050_*.mat'.format(experiment))
# data = {'X':[]}
data = {'X':[],'P':[]}
for f in sorted(npzs):
    s = loadmat(f)
    print(f,len(s['X']))
    data['X'].extend(s['X'])
    data['P'].extend(s['P'])
savemat(opath,data)
print('save to', opath, len(data['X']))