#%%
%load_ext autoreload
%autoreload 2
#%%
from pathlib import Path
from diffome.connectome.base import Connectome
from diffome.tda.barcode import BarCode
import nibabel as nib
import os

from dipy.data.fetcher import (fetch_file_formats,
                               get_file_formats)

def pull_dipy_trk():
    fetch_file_formats()
    bundles_filename, ref_anat_filename = get_file_formats()
    reference_anatomy = nib.load(ref_anat_filename)

    return bundles_filename[0], reference_anatomy

#%%
input_trk_left: Path = "/home/virati/Data/postdoc/connectome_transfer/petersen_top/petersen_pd_top_left.trk"
input_trk_right: Path = "/home/virati/Data/postdoc/connectome_transfer/petersen_top/petersen_pd_top_right.trk"
ref_anat_filename = 'same'

petersen_subatlas_left = (input_trk_left, ref_anat_filename)
petersen_subatlas_right = (input_trk_right, ref_anat_filename)
tract_list = {'left': petersen_subatlas_left, 'right': petersen_subatlas_right,}

barcode_stack = {key: [] for key in tract_list}
for key, val in tract_list.items():
    main_connectome = Connectome(*val)
    for ii in range(100):
        main_connectome.subsample(factor=500)
        barcode_analysis = BarCode(main_connectome)
        barcode_analysis.calculate(do_plot = False, ignore_streamlines=[ii])
        barcode_stack[key].append(barcode_analysis.barcode)

#%%
aggregate_barcode = {}
import matplotlib.pyplot as plt
for side in ['left','right']:
    aggregate_barcode[side] = [[barcode_stack[side][iter][element][1] for element in range(len(barcode_stack[side][iter]))] for iter in range(10)]
    aggregate_barcode[side] = [item for sublist in aggregate_barcode[side] for item in sublist]
    xval = [item[0] for item in aggregate_barcode[side]]
    yval = [item[1] for item in aggregate_barcode[side]]
    plt.scatter(xval, yval,alpha=0.5)

#%%
# calculate distance between the aggregate persistence
from diffome.tda.wasser import persistence_wasserstein
