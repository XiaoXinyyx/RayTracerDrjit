import drjit as dr

class Optimizer:
    def __init__(self):
        pass

    def zero_grad(self):
        pass

    def step(self):
        pass

class SGD(Optimizer):
    def __init__(self, lr: float, params: list):
        self.lr = lr
        self.params = []
        if isinstance(params, list):
            for param in params:
                dr.enable_grad(param)
                self.params.append(param)
        else:
            raise TypeError("Invalid parameter type")

    def zero_grad(self):
        for param in self.params:
            dr.disable_grad(param)
            dr.enable_grad(param)
            #dr.clear_grad(param)

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad