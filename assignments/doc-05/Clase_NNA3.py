import numpy as np

nn_0 = 2
nn_1 = 2
nn_2 = 1

# Sesgos y pesos
b1 = np.array([[0.45], [0.35]])
W1 = np.array([[0.15, 0.20], [0.25, 0.30]])
b2 = np.array([[0.30]])
W2 = np.array([[0.35, 0.48]])

# Vectores de entrada y salida
Z1 = np.zeros((nn_1,1))
A1 = np.zeros((nn_1,1))
Z2 = np.zeros((nn_2,1))
A2 = np.zeros((nn_2,1))

# Función de propagación hacia adelante
def propaga(Xo):
    Z1 = np.dot(W1,Xo) + b1
    A1 = sig(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sig(Z2)
    return A2, Z2, A1, Z1

# Función de activación de sigmoide
def sig(s):
    return 1/(1+np.exp(-s))

def dSig(s):
    df = sig(s.T[0])*(1 - sig(s.T[0]))
    ds = np.diag(df)
    return ds

# Valores originales
# X = np.array([[0.05], [0.10]])
# Yr = np.array([[0.01], [0.99]])

# Nuevo X
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

# Yr para NAND
# Yr = np.array([[1, 1, 1, 0]])

# Yr para XOR
Yr = np.array([[0, 1, 1, 0]])

# A2, Z2, A1, Z1 = propaga(X)
# print(A2)

def error(A2, Yd):
    err = Yd - A2
    return err

def backpropagation(X, Yr, A2, Z2, A1, Z1):
    delta2 = np.dot(dSig(Z2), error(A2, Yr))
    dEdW2 = -np.dot(delta2, A1.T) # Corrección para W2
    dEdb2 = -delta2

    delta1 = np.dot(np.dot(dSig(Z1), W2.T), delta2)
    dEdW1 = -np.dot(delta1,X.T) # Correccion para W1

    dEdb1 = -delta1
    return dEdW2, dEdb2, dEdW1, dEdb1

# dEdW2, dEdb2, dEdW1, dEdb1 = backpropagation(X, Yr, A2, Z2, A1, Z1)

eta = 0.1

# W2 = W2 - eta * dEdW2
# b2 = b2 - eta * dEdb2
# W1 = W1 - eta * dEdW1
# b1 = b1 - eta * dEdb1

# A2, Z2, A1, Z1 = propaga(X)
# print(A2)

i = 0
for i in range(50000):

    for j in range(4):

        Xo = X[:,j].reshape(2,1)
        Yd = Yr[:,j].reshape(1,1)

        A2, Z2, A1, Z1 = propaga(Xo)

        dEdW2, dEdb2, dEdW1, dEdb1 = backpropagation(Xo, Yd, A2, Z2, A1, Z1)

        W2 = W2 - eta * dEdW2
        b2 = b2 - eta * dEdb2
        W1 = W1 - eta * dEdW1
        b1 = b1 - eta * dEdb1

# print(A2)

for j in range(4):
    Xo = X[:,j].reshape(2,1)
    A2,_,_,_ = propaga(Xo)
    print(X[:,j], A2)