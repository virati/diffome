from dipy.viz import actor, window

BUNDLE_COLOR = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]


class ConnectomeRenderer:
    def __init__(self, connectome_list=None):
        if connectome_list is None:
            connectome_list = []
        self.connectome_list = connectome_list

    def render(self, color_per_bundle=False):
        scene = window.Scene()

        for cc, connectome in enumerate(self.connectome_list):
            if color_per_bundle:
                stream_actor = actor.line(
                    connectome.streamlines.streamlines, BUNDLE_COLOR[cc]
                )
            else:
                stream_actor = actor.line(connectome.streamlines.streamlines)

            scene.add(stream_actor)

        # Uncomment the line below to show to display the window
        window.show(scene, size=(600, 600), reset_camera=False)
