#%%
%load_ext autoreload
%autoreload 2
#%%
from pathlib import Path
from diffome.connectome.base import Connectome
from diffome.tda.barcode import BarCode
import nibabel as nib
import os

#%%
input_trk_paths = {
    "petersen": {
        "left": ("/home/virati/Data/postdoc/connectome_transfer/petersen_top/petersen_pd_top_left.trk", "same"),
        "right": ("/home/virati/Data/postdoc/connectome_transfer/petersen_top/petersen_pd_top_right.trk", "same"),
    },
    "hcp": {
        "left": ("/home/virati/Data/postdoc/subatlases/july_run_hcp_bilat/july_run_hcp_bilat_100_mirrorPD_top_left.trk", "same"),
        "right": ("/home/virati/Data/postdoc/subatlases/july_run_hcp_bilat/july_run_hcp_bilat_100_mirrorPD_top_right.trk", "same"),
    },
    "mgh": {
        "left": ("/home/virati/Data/postdoc/subatlases/july_run_mgh_bilat/july_run_mgh_bilat_100_mirrorPD_top_left.trk", "same"),
        "right": ("/home/virati/Data/postdoc/subatlases/july_run_mgh_bilat/july_run_mgh_bilat_100_mirrorPD_top_right.trk", "same"),
    },
}

#%%#
ref_anat_filename = 'same'
connectome_name = 'petersen'
#petersen_subatlas_left = (input_trk_left, ref_anat_filename)
#petersen_subatlas_right = (input_trk_right, ref_anat_filename)
#tract_list = {'left': petersen_subatlas_left, 'right': petersen_subatlas_right,}

tract_list = input_trk_paths[connectome_name]

connectomes = [Connectome(*val).clip_streamlines(n_clip=100) for val in tract_list.values()]

#%%
from diffome.tda.compare import TDAComparison
TDA_comp = TDAComparison(connectomes)
TDA_comp.calculate()
TDA_comp.aggregate_barcodes().plot_aggregate_barcodes()
#%%
import matplotlib.pyplot as plt
for stack in range(2):
    TDA_comp.calculate_distance_distributions_inside(which_stack=stack, do_plot=True, hold_plot=True)
plt.plot()
#%%
# final calculation
TDA_comp.calculate_cross_distance()
#%%