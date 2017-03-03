import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions, nopredictions = False):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """
    N = data.shape[0]
    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    z1 = data.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    prob = softmax(a2)
    cost = - np.sum(np.log(np.sum(labels * prob, axis = 1, keepdims = True)))
    cost /= N
    pred = np.argmax(prob, axis = 1)
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    corr = np.zeros(prob.shape)
    sig_grad2 = sigmoid_grad(a2)
    sig_grad1 = sigmoid_grad(a1)
    delta2 = (prob - labels) * sig_grad2
    delta1 = delta2.dot(W2.T) * sig_grad1

    gradW1 = data.T.dot(delta1)/N
    gradb1 = np.sum(delta1, axis = 0, keepdims = True)/N
    gradW2 = a1.T.dot(delta2)/N
    gradb2 = np.sum(delta2, axis = 0, keepdims = True)/N
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    
    if nopredictions:
        return cost, grad
    else:
        return cost, grad, pred

def neural_wrapper(data, labels, params, dimensions):
    cost, grad, _ = forward_backward_prop(data, labels, params, dimensions)
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions, nopredictions = True), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()