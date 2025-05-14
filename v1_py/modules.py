import numpy as np

class MM:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.W = np.random.randn(self.out_dim, self.in_dim) * 0.01
        self.b = np.random.rand(out_dim, 1)

        self.layer_type = "MM"

    def forward(self, z):
        return np.matmul(self.W, z) + self.b
    
    def backward(self, v):
        return np.matmul(self.W.T, v)
    
    def update_w_b(self, z, v, lr):
        dw = np.matmul(v, z.T)
        db = v

        self.W = self.W - (lr * dw)
        self.b = self.b - (lr * db)

class ReLU:
    def __init__(self, in_dim):
        self.in_dim = in_dim
        self.layer_type = "ReLU"

    def forward(self, z):
        return np.maximum(z, 0)
    
    def backward(self, z, v):
        dz = np.zeros(z.shape)
        for q in range(z.shape[0]):
            if(z[q, 0] > 0):
                dz[q, 0] = v[q, 0]
        
        return dz

def softmax(z):
    z = z.reshape(-1, 1)
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z, axis=0, keepdims=True)

class CE_loss:
    def __init__(self, in_dim):
        self.in_dim = in_dim

    def forward(self, z, y):
        SM_z = softmax(z)
        return -np.sum(y * np.log(SM_z + 1e-10))
    
    def backward(self, z, y):
        return softmax(z) - y
    
class NN:
    def __init__(self, layers, loss, lr):
        self.layers = layers
        self.loss = loss
        self.lr = lr

    def train(self, x, y):
        outputs = [x]
        for layer in self.layers:
            outputs.append(layer.forward(outputs[-1]))
        
        J = self.loss.forward(outputs[-1], y)

        backwards = [self.loss.backward(outputs[-1], y)]
        self.layers[-1].update_w_b(outputs[-2], backwards[-1], self.lr)
        backwards.append(self.layers[-1].backward(backwards[-1]))

        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            
            if layer.layer_type == "MM":
                layer.update_w_b(outputs[i], backwards[-1], self.lr)
                backwards.append(layer.backward(backwards[-1]))
                
            
            if layer.layer_type == "ReLU":
                backwards.append(layer.backward(outputs[i], backwards[-1]))

        return J
    
    def predict_proba(self, x):
        input = x
        for layer in self.layers:
            input = layer.forward(input)

        return softmax(input)
    
    def predict(self, x):
        input = x
        for layer in self.layers:
            input = layer.forward(input)

        return np.argmax(input)