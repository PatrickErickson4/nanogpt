import torch
class LayerNorm:
    
    def __init__(self, dim, epsilon = 1e-5, momentum = .1):

        self.epsilon = epsilon
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):

        xMean = x.mean(1, keepdim=True)
        xVar = x.var(1, keepdim=True)

        xHat = (x - xMean) / torch.sqrt(xVar + self.epsilon)
        self.toRet = self.gamma * xHat + self.beta

    def parameters(self):
        return [self.gamma, self.beta]