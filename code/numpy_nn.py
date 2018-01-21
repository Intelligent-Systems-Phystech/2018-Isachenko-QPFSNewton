import numpy as np

class Module(object):
    def __init__ (self):
        self.output = None
        self.gradInput = None
        self.train = True
    
    def forward(self, input):
        return self.updateOutput(input)

    def backward(self, input, gradOutput):
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput

    def updateOutput(self, input):
        pass

    def updateGradInput(self, input, gradOutput):
        pass   
    
    def accGradParameters(self, input, gradOutput):
        pass
    
    def zeroGradParameters(self): 
        pass
        
    def getParameters(self):
        return []
        
    def getGradParameters(self):
        return []
    
    def training(self):
        self.train = True
    
    def evaluate(self):
        self.train = False
    
    def __repr__(self):
        return "Module"


class Sequential(Module):
    
    def __init__ (self):
        super(Sequential, self).__init__()
        self.modules = []
   
    def add(self, module):
        self.modules.append(module)

    def updateOutput(self, input):
        n = len(self.modules)
        layer_outputs = [input, self.modules[0].forward(input)]
        
        for i in range(1, n):
            layer_outputs.append(self.modules[i].forward(layer_outputs[-1]))
        
        self.layer_outputs = layer_outputs
        self.output = layer_outputs[-1]
        
        return self.output

    def backward(self, input, gradOutput):
        n = len(self.modules)
        g = self.modules[-1].backward(self.layer_outputs[-2], gradOutput)
        
        for i in range(2, n + 1):
            g = self.modules[-i].backward(self.layer_outputs[-(i + 1)], g)
        
        self.gradInput = g
        
        return self.gradInput
      

    def zeroGradParameters(self): 
        for module in self.modules:
            module.zeroGradParameters()
    
    def getParameters(self):
        return [x.getParameters() for x in self.modules]
    
    def getGradParameters(self):
        return [x.getGradParameters() for x in self.modules]
    
    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string
    
    def __getitem__(self,vx):
        return self.modules.__getitem__(x)


class Linear(Module):
    def __init__(self, n_in, n_out, W_init=None, b_init=None):
        super(Linear, self).__init__()
       
        # This is a nice initialization
        stdv = 1./np.sqrt(n_in)
        if W_init is not None:
            assert W_init.shape == (n_out, n_in), "The weights have not appropriate shape"
            self.W = W_init
        else:
            self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in))
        if b_init is not None:
            assert b_init.shape == (n_out,), "The biases have not appropriate shape"
            self.b = b_init
        else:
            self.b = np.random.uniform(-stdv, stdv, size = n_out)
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def updateOutput(self, input):
        self.output = input.dot(self.W.T) + self.b
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput.dot(self.W)
        
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        self.gradW = gradOutput.T.dot(input)
        self.gradb = np.sum(gradOutput, axis=0)
    
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Linear(%d -> %d)' %(s[1],s[0])
        return q


class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        self.output = np.exp(self.output) / np.exp(self.output).sum(axis=1, keepdims=True)
        
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(np.diagonal(gradOutput.dot(self.output.T))[:, np.newaxis],
                                     self.output)
        self.gradInput = np.multiply(gradOutput, self.output) - self.gradInput
        return self.gradInput
    
    def __repr__(self):
        return "SoftMax"


class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput , input > 0)
        return self.gradInput
    
    def __repr__(self):
        return "ReLU"
    
    
class LeakyReLU(Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()
        self.slope = slope
        
    def updateOutput(self, input):
        self.output = np.maximum(input, 0) + self.slope * np.minimum(0, input)
        
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput , input > 0) + \
                            self.slope * np.multiply(gradOutput , input < 0)
        return self.gradInput
    
    def __repr__(self):
        return "LeakyReLU(slope={:.3f})".format(self.slope)
    
    
class ELU(Module):
    def __init__(self, alpha = 1.0):
        super(ELU, self).__init__()
        self.alpha = alpha
        
    def updateOutput(self, input):
        self.output = np.maximum(input, 0) + \
                        np.minimum(0, self.alpha * (np.exp(input) - 1) * input)
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput , input > 0) + \
                            self.alpha * np.multiply(np.exp(input), 
                                                     np.multiply(gradOutput , input < 0))
        return self.gradInput
    
    def __repr__(self):
        return "ELU"
    

class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.log(1 + np.exp(input))
        
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, 1. / (1 + np.exp(-input)))
        return self.gradInput
    
    def __repr__(self):
        return "SoftPlus"


class Criterion(object):
    def __init__ (self):
        self.output = None
        self.gradInput = None
        
    def forward(self, input, target):
        return self.updateOutput(input, target)

    def backward(self, input, target):
        return self.updateGradInput(input, target)
    
    def updateOutput(self, input, target):
        return self.output

    def updateGradInput(self, input, target):
        return self.gradInput   

    def __repr__(self):
        return "Criterion"


class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()
        
    def updateOutput(self, input, target):   
        self.output = np.sum(np.power(input - target, 2)) / input.shape[0]
        return self.output 
 
    def updateGradInput(self, input, target):
        self.gradInput  = (input - target) / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"


class ClassNLLCriterion(Criterion):
    def __init__(self):
        a = super(ClassNLLCriterion, self)
        super(ClassNLLCriterion, self).__init__()
        
    def updateOutput(self, input, target): 
        eps = 1e-15 
        input_clamp = np.clip(input, eps, 1 - eps)
        n = input.shape[0]
        self.output = - np.sum(np.multiply(target, np.log(input_clamp))) / n
        
        return self.output

    def updateGradInput(self, input, target):
        input_clamp = np.maximum(1e-15, np.minimum(input, 1 - 1e-15) )
        n = input.shape[0]
        self.gradInput = - np.divide(target, input_clamp) / n
        
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterion"