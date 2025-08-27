#%%
%load_ext autoreload
%autoreload 2
#%%
from pathlib import Path
from diffome.connectome.base import Connectome

from dipy.data import fetch_bundle_atlas_hcp842

from diffome.viz import ConnectomeRenderer
from diffome.tda.compare import TDAComparison

if False:
    files, folder = fetch_bundle_atlas_hcp842()
trk_dir = Path('/home/virati/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/bundles')
trks = {f.stem: f for f in trk_dir.glob('*.trk')}

#%%
big_connectome = []

#do_fibers = ['ICP_R','ICP_L']
#do_fibers = ['MLF_R','MLF_L']
do_fibers = ['UF_L','UF_R']
#do_fibers = ['UF_L', 'ICP_L']
#do_fibers = trks.keys()
for fiber_name in do_fibers:
    main_connectome = Connectome(str(trks[fiber_name]),"same", bbox_valid_check=False)
    big_connectome.append(main_connectome)

#%%
# Do basic TDA stuff here
connectomes = [Connectome(str(trks[val]), "same", bbox_valid_check=False).clip_streamlines(n_clip=100) for val in do_fibers]
TDA_comp = TDAComparison(connectomes)
TDA_comp.calculate(downsample_points=200)
#%%
TDA_comp.aggregate_barcodes().plot_aggregate_barcodes()

plot_together = True

for stack in range(2):
    TDA_comp.calculate_distance_distributions_inside(which_stack=stack, do_plot=True, hold_plot=plot_together)
#%%
if plot_together:
    import matplotlib.pyplot as plt
    plt.legend(do_fibers)
    plt.plot()

## Final Cross Distance
TDA_comp.calculate_cross_distance()
#%%
# Do Rendering
full_render = ConnectomeRenderer(big_connectome)
full_render.render()
