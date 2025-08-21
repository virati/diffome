class TDAAnalysis:
    def __init__(self, input_connectome: Connectome):
        self.input_connectome = input_connectome


class BarCode(TDAAnalysis):
    def __init__(self, input_connectome: Connectome):
        super().__init__(input_connectome)

    def calculate(self, params: dict = None) -> None:
        return self

    def plot_barcode(self):
        pass
