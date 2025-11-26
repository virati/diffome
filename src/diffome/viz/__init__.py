from dipy.viz import actor, window
import matplotlib.cm as cm
from typing import List
from nilearn.datasets import load_mni152_template

# BUNDLE_COLOR = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]
BUNDLE_COLOR = cm.get_cmap("tab10")


class ConnectomeRenderer:
    def __init__(self, connectome_list: List = None):
        if connectome_list is None:
            connectome_list = []
        self.connectome_list = connectome_list
        self.scene = window.Scene()

    def render(
        self, color_per_bundle=False, do_connectomes=None, alpha=1.0, show_template=True
    ):
        scene = self.scene

        if show_template:
            mni_t1_template = load_mni152_template()
            mni_data = mni_t1_template.get_fdata()
            mni_affine = mni_t1_template.affine

            template_z_slice = actor.slicer(mni_t1_template.dataobj, affine=mni_affine)
            scene.add(template_z_slice)

        for cc, connectome in enumerate(self.connectome_list):
            if cc not in do_connectomes:
                continue
            if color_per_bundle:
                stream_actor = actor.line(
                    connectome.streamlines.streamlines, BUNDLE_COLOR(cc)
                )
            else:
                stream_actor = actor.line(
                    connectome.streamlines.streamlines, opacity=alpha
                )

            scene.add(stream_actor)

        # Uncomment the line below to show to display the window
        # window.show(scene, size=(600, 600), reset_camera=False)
        self.scene = scene
        return self
