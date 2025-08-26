#%%
%load_ext autoreload
%autoreload 2
#%%
from pathlib import Path
from diffome.connectome.base import Connectome
from diffome.tda.barcode import BarCode
import nibabel as nib
import os

from dipy.data import fetch_bundle_atlas_hcp842

from diffome.viz import ConnectomeRenderer

files, folder = fetch_bundle_atlas_hcp842()
#%%
#trks = {'a': Path('/home/virati/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/bundles/AC.trk')}
#sanity
#main_connectome = Connectome(str(trks['a']),"same", bbox_valid_check=False)
#main_connectome.subsample(1).render()
#%%
trk_dir = Path('/home/virati/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/bundles')
trks = {f.stem: f for f in trk_dir.glob('*.trk')}

#%%
big_connectome = []
do_fibers = ['MLF_R','MLF_L']
for fiber_name in do_fibers:
    main_connectome = Connectome(str(trks[fiber_name]),"same", bbox_valid_check=False)
    big_connectome.append(main_connectome)

#%%
# Do basic TDA stuff here

#%%
full_render = ConnectomeRenderer(big_connectome)
full_render.render()
