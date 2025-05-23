import numpy as np

class MM:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # He initalization for ReLU
        self.W = np.random.randn(out_dim, in_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros((out_dim, 1))

        self.layer_type = "MM"

    def forward(self, Z):
        return np.matmul(self.W, Z) + self.b # ok
    
    # ok
    def backward(self, v):
        return np.matmul(self.W.T, v)
    
    # I believe this is ok
    def update_w_b(self, a, v, lr):
        dw = np.matmul(v, a.T)
        db = np.sum(v, axis=1, keepdims=True)

        self.W = self.W - (lr * dw)
        self.b = self.b - (lr * db)

class ReLU:
    def __init__(self, in_dim):
        self.in_dim = in_dim
        self.layer_type = "ReLU"

    def forward(self, z):
        return np.maximum(z, 0) # ok
    
    # changed, ok?
    def backward(self, z, v):
        mask = z > 0
        return v * mask
        
def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    e_Z = np.exp(Z_shifted)
    return e_Z / np.sum(e_Z, axis=0, keepdims=True)

class CE_loss:
    def __init__(self, in_dim):
        self.in_dim = in_dim

    def forward(self, Z, y):
        SM_z = softmax(Z)
        return -np.sum(y * np.log(SM_z + 1e-10)) # ok
    
    def backward(self, z, y):
        return softmax(z) - y # ok
    
class NN_minimal:
    def __init__(self, layers, loss, lr):
        self.layers = layers
        self.loss = loss
        self.lr = lr

    def train(self, x, y):
        forwards = [x]
        for layer in self.layers:
            forwards.append(layer.forward(forwards[-1]))

        last_layer_output = forwards[-1]
        J = self.loss.forward(last_layer_output, y)

        backwards = [self.loss.backward(last_layer_output, y)]

        k = len(forwards) - 2
        for layer in list(reversed(self.layers)):
            if layer.layer_type == "MM":
                grad_z = layer.backward(backwards[-1])
                layer.update_w_b(forwards[k], backwards[-1], self.lr)
                backwards.append(grad_z)
            elif layer.layer_type == "ReLU":
                backwards.append(layer.backward(forwards[k], backwards[-1]))
            k -= 1

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

        W = input
        y_hat = np.zeros(W.shape)

        for i in range(W.shape[1]):
            y_hat[np.argmax(W[:, i:i+1]), i] = 1

        return y_hat