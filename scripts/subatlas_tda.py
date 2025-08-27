#%%
%load_ext autoreload
%autoreload 2
#%%
from diffome.connectome.base import Connectome
from diffome.tda.compare import TDAComparison
from diffome.viz import ConnectomeRenderer

input_trk_paths = {
    "petersen": {
        "left": ("/home/virati/Data/postdoc/connectome_transfer/petersen_top/petersen_pd_top_left.trk", "same"),
        "right": ("/home/virati/Data/postdoc/connectome_transfer/petersen_top/petersen_pd_top_right.trk", "same"),
    },
    "petersen100":{
        "right": ("/home/virati/Data/postdoc/subatlases/july_develop_petersen/july_develop_petersen_100PD_top_right.trk", "same")
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

#%%
tract_list = input_trk_paths[connectome_name]
tract_list = ["/home/virati/Data/postdoc/subatlases/july_develop_petersen/july_develop_petersen_100PD_top_right.trk", "/home/virati/Data/postdoc/subatlases/july_run_mgh_bilat/july_run_mgh_bilat_100_mirrorPD_top_right.trk",]
connectomes = [Connectome(val, "same").clip_streamlines(n_clip=100) for val in tract_list]
TDA_comp = TDAComparison(connectomes)

TDA_comp.calculate()
TDA_comp.aggregate_barcodes().plot_aggregate_barcodes()
#%%
plot_together = True

for stack in range(2):
    TDA_comp.calculate_distance_distributions_inside(which_stack=stack, do_plot=True, hold_plot=plot_together)

if plot_together:
    import matplotlib.pyplot as plt
    plt.plot()
#%%
# final calculation
TDA_comp.calculate_cross_distance()
#%%
full_render = ConnectomeRenderer(connectomes)
full_render.render(color_per_bundle=True)