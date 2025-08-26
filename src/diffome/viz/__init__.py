from dipy.viz import actor, window


class ConnectomeRenderer:
    def __init__(self, connectome_list=None):
        if connectome_list is None:
            connectome_list = []
        self.connectome_list = connectome_list

    def render(self):
        scene = window.Scene()

        for connectome in self.connectome_list:
            stream_actor = actor.line(connectome.streamlines.streamlines)

            scene.add(stream_actor)

        # Uncomment the line below to show to display the window
        window.show(scene, size=(600, 600), reset_camera=False)
