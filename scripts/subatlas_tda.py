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
#%%
TDA_comp.aggregate_barcodes().plot_aggregate_barcodes().calculate_distance_distributions_inside(do_plot=True)
#%%
# OLD

jack_num = 100
barcode_stack = {key: [] for key in tract_list}
barcode_stack_culprit = {key: [] for key in tract_list}
for key, val in tract_list.items():
    main_connectome = Connectome(*val)
    for ii in range(jack_num):
        main_connectome.clip_streamlines(n_clip=jack_num)
        barcode_analysis = BarCode(main_connectome)
        barcode_analysis.calculate(do_plot = False, ignore_streamlines=[ii], downsample_points=100)
        barcode_stack[key].append(barcode_analysis.barcode)
        barcode_stack_culprit[key].append(ii)
        #add in the index itself here, so we can start doing streamline attribution?

#%%
aggregate_barcode = {}
import matplotlib.pyplot as plt
for side in ['left','right']:
    aggregate_barcode[side] = [[barcode_stack[side][iter][element][1] for element in range(len(barcode_stack[side][iter]))] for iter in range(jack_num)]
    aggregate_barcode[side] = [item for sublist in aggregate_barcode[side] for item in sublist]
    xval = [item[0] for item in aggregate_barcode[side]]
    yval = [item[1] for item in aggregate_barcode[side]]
    plt.scatter(xval, yval,alpha=0.05)
    plt.show()

#%%
# calculate distance between the aggregate persistence
#from diffome.tda.wasser import persistence_wasserstein
import numpy as np
from gudhi.wasserstein import wasserstein_distance as wass_dist

#first_barcode = np.array([b for a,b in barcode_analysis.barcode])
#second_barcode = np.array([b for a,b in barcode_analysis.barcode])

# let's check inside each side first
sides = ['left','right']
intra_dist = {key: [] for key in ['left', 'right']}
for side in ['left','right']:
    for ii in range(len(barcode_stack[side])):
        first_barcode = np.array([b for a,b in barcode_stack[side][ii]])
        for jj in range(ii+1, len(barcode_stack[side])):
            if ii == jj:
                continue
            second_barcode = np.array([b for a,b in barcode_stack[side][jj]])
            test = wass_dist(first_barcode, second_barcode)
            intra_dist[side].append(test)
#%%
for side in sides:
    plt.hist(intra_dist[side], bins=30, alpha=0.5)
#%%
# Comparing Across Sides
first_barcode = np.array(aggregate_barcode['left'])
second_barcode = np.array(aggregate_barcode['right'])
test = wass_dist(first_barcode, second_barcode)
print(test)

#%%
#Inter Hemi Distances
inter_dist = []
for ii in range(len(barcode_stack['left'])):
    first_barcode = np.array([b for a,b in barcode_stack['left'][ii]])
    for jj in range(ii+1, len(barcode_stack['right'])):
        if ii == jj:
            continue
        second_barcode = np.array([b for a,b in barcode_stack['right'][jj]])
        test = wass_dist(first_barcode, second_barcode)
        inter_dist.append(test)

plt.hist(inter_dist, bins=30, alpha=0.5)