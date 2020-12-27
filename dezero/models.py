from dezero.layers import Layer
from dezero import utils


class Model(Layer):
    def plot(self, *inputs, to_file="model.png"):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=False, to_file=to_file)
