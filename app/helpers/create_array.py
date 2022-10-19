import numpy as np
import glob
import pandas as pd
import os
from datetime import datetime

paths = list(sorted(glob.glob('/home/marten/Desktop/workdir/bkp_gan/data_raw//2*/*/*.npy')))
#print('/home/marten/Desktop/workdir/bkp_gan/data_raw//2*/*/*.npy')

def pre_process_imgs(path):
    image = np.load(path)
    dim_0, dim_1 = image.shape[0], image.shape[1]
    image = image.reshape(dim_0*dim_1)
    noise = np.random.rand(dim_0*dim_1)
    input_array = np.stack([noise,image], axis=0)
    return input_array.reshape(2,dim_0,dim_1)

X = []
dates = []
for i, path in enumerate(list(sorted(paths))):
    datestr = os.path.basename(path).split('_')[-1].split('.')[0]
    dateobj = datetime.strptime(datestr, "%Y%m%d%H")
    batch = pre_process_imgs(path)
    print(batch.shape)
    #if i > 507:

    if (batch.shape[1] == 831) and (batch.shape[2] == 1150):
        X.append(batch)
        dates.append(datestr)
        X_array = np.array(X)
        print(X_array.shape)

        if X_array.shape[0] == 10:
            path_out = f"/home/marten/Desktop/workdir/gans/data/{dateobj.strftime('%Y/%j')}/"
            os.makedirs(path_out, exist_ok=True)
            pd.DataFrame({'dates':dates}).to_csv(f"{path_out}{datestr}.csv")
            np.save(f"{path_out}{datestr}", X_array)
            X = []
            dates = []