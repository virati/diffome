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

input_trk: Path = "/home/virati/Data/postdoc/connectome_transfer/petersen_top/petersen_pd_top_left.trk"
ref_anat_filename = None


def pull_dipy_trk():
    fetch_file_formats()
    bundles_filename, ref_anat_filename = get_file_formats()
    reference_anatomy = nib.load(ref_anat_filename)

    return bundles_filename[0], reference_anatomy

main_connectome = Connectome(*pull_dipy_trk())
for ii in range(10):
    main_connectome.subsample(factor=1000)
    barcode_analysis = BarCode(main_connectome)
    barcode_analysis.calculate(do_plot = True)
