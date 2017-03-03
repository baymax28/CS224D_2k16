import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

from cs224d.data_utils import *

from q3_sgd import load_saved_params, sgd
from q4_softmaxreg import softmaxRegression, getSentenceFeature, accuracy, softmax_wrapper
from q2_neural import forward_backward_prop, neural_wrapper

# Try different regularizations and pick the best!
# NOTE: fill in one more "your code here" below before running!
REGULARIZATION = None   # Assign a list of floats in the block below
### YOUR CODE HERE
REGULARIZATION = [0]#, 1e-6, 1e-5, 1e-4, 1e-3]
ITER = 500000
### END YOUR CODE

# Load the dataset
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# Load the word vectors we trained earlier 
_, wordVectors0, _ = load_saved_params()
wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])
dimVectors = wordVectors.shape[1]

# Load the train set
trainset = dataset.getTrainSentences()
nTrain = len(trainset)
# trainFeatures = np.zeros((nTrain, dimVectors))
# trainLabels = np.zeros((nTrain,), dtype=np.int32)
# for i in xrange(nTrain):
#     words, trainLabels[i] = trainset[i]
#     trainFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)
##########################################################################
SMALLSIZE = 100
idx = np.random.choice(nTrain, SMALLSIZE)
trainsetArray = np.asarray(trainset)
smallTrainset = trainsetArray[idx]
trainFeatures = np.zeros((SMALLSIZE, dimVectors))
trainLabels = np.zeros((SMALLSIZE,), dtype=np.int32)
for i in xrange(SMALLSIZE):
    words, trainLabels[i] = smallTrainset[i]
    trainFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)
##########################################################################
print "Training set size : %d" % trainFeatures.shape[0]

# Prepare dev set features
devset = dataset.getDevSentences()
nDev = len(devset)
devFeatures = np.zeros((nDev, dimVectors))
devLabels = np.zeros((nDev,), dtype=np.int32)
for i in xrange(nDev):
    words, devLabels[i] = devset[i]
    devFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

print "Development set size : %d" % devFeatures.shape[0]

# Try our regularization parameters
results = []
for regularization in REGULARIZATION:
    random.seed(3141)
    np.random.seed(59265)
    ###########################################

    print "Training with siftmax regression!"

    # weights = np.random.randn(dimVectors, 5)
    # print "Training for reg=%f" % regularization 

    # # We will do batch optimization
    # weights = sgd(lambda weights: softmax_wrapper(trainFeatures, trainLabels, 
    #     weights, regularization), weights, 3.0, ITER, PRINT_EVERY=100)
    
    # # Test on train set
    # _, _, pred = softmaxRegression(trainFeatures, trainLabels, weights)
    # trainAccuracy = accuracy(trainLabels, pred)
    # print "Train accuracy (%%): %f" % trainAccuracy

    # # Test on dev set
    # _, _, pred = softmaxRegression(devFeatures, devLabels, weights)
    # devAccuracy = accuracy(devLabels, pred)
    # print "Dev accuracy (%%): %f" % devAccuracy

    #################### Training with 2 layer neural net. Not working. #######################
    #################### Cost function getting stagnated at high value. #######################

    print "Training with 2 layer neural net!"

    dimensions = [dimVectors, 50, 5]
    weights = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )
    tr_N = trainLabels.shape[0]
    tr_labels = np.zeros((tr_N, dimensions[2]))
    for i in xrange(tr_N):
        tr_labels[i, trainLabels[i]] = 1

    dv_N = devLabels.shape[0] 
    dv_labels = np.zeros((dv_N, dimensions[2]))
    for i in xrange(dv_N):
        dv_labels[i, devLabels[i]] = 1

    weights = sgd(lambda weights: neural_wrapper(trainFeatures, tr_labels, weights, dimensions), 
        weights, 3.0, ITER, PRINT_EVERY=100)

    _, _, pred = forward_backward_prop(trainFeatures, tr_labels, weights, dimensions)
    trainAccuracy = accuracy(trainLabels, pred)
    print "Train accuracy (%%): %f" % trainAccuracy

    _, _, pred = forward_backward_prop(devFeatures, dv_labels, weights, dimensions)
    devAccuracy = accuracy(devLabels, pred)
    print "Dev accuracy (%%): %f" % devAccuracy
    
    ###########################################

    # Save the results and weights
    results.append({
        "reg" : regularization, 
        "weights" : weights, 
        "train" : trainAccuracy, 
        "dev" : devAccuracy})

# Print the accuracies
print ""
print "=== Recap ==="
print "Reg\t\tTrain\t\tDev"
for result in results:
    print "%E\t%f\t%f" % (
        result["reg"], 
        result["train"], 
        result["dev"])
print ""

# Pick the best regularization parameters
BEST_REGULARIZATION = None
BEST_WEIGHTS = None     

### YOUR CODE HERE 
BEST_REGULARIZATION = max(results,key=lambda item:item["dev"])["reg"]
BEST_WEIGHTS = max(results,key=lambda item:item["dev"])["weights"]
### END YOUR CODE

# Test your findings on the test set
# testset = dataset.getTestSentences()
# nTest = len(testset)
# testFeatures = np.zeros((nTest, dimVectors))
# testLabels = np.zeros((nTest,), dtype=np.int32)
# for i in xrange(nTest):
#     words, testLabels[i] = testset[i]
#     testFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

# _, _, pred = softmaxRegression(testFeatures, testLabels, BEST_WEIGHTS)
# print "Best regularization value: %E" % BEST_REGULARIZATION
# print "Test accuracy (%%): %f" % accuracy(testLabels, pred)

# Make a plot of regularization vs accuracy
# plt.plot(REGULARIZATION, [x["train"] for x in results])
# plt.plot(REGULARIZATION, [x["dev"] for x in results])
# # plt.xscale('log')
# plt.xlabel("regularization")
# plt.ylabel("accuracy")
# plt.legend(['train', 'dev'], loc='upper left')
# plt.savefig("q4_reg_v_acc.png")
# plt.show()

