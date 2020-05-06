import numpy as np


class LSTM:
    # LSTM cell (input, output, amount of recurrence, learning rate)
    def __init__(self, xs, ys, rl, lr):
        # input is word length x word length
        self.x = np.zeros(xs + ys)
        # input size is word length + word length
        self.xs = xs + ys
        # output
        self.y = np.zeros(ys)
        # output size
        self.ys = ys
        # cell state intialized as size of prediction
        self.cs = np.zeros(ys)
        # how often to perform recurrence
        self.rl = rl
        # balance the rate of training (learning rate)
        self.lr = lr
        # init weight matrices for our gates
        # forget gate
        self.f = np.random.random((ys, xs + ys))
        # input gate
        self.i = np.random.random((ys, xs + ys))
        # cell state
        self.c = np.random.random((ys, xs + ys))
        # output gate
        self.o = np.random.random((ys, xs + ys))
        # forget gate gradient
        self.Gf = np.zeros_like(self.f)
        # input gate gradient
        self.Gi = np.zeros_like(self.i)
        # cell state gradient
        self.Gc = np.zeros_like(self.c)
        # output gate gradient
        self.Go = np.zeros_like(self.o)

    # activation function to activate our forward prop, just like in any type of neural network
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid to help computes gradients
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # tanh! another activation function, often used in LSTM cells
    # Having stronger gradients: since data is centered around 0,
    # the derivatives are higher. To see this, calculate the derivative
    # of the tanh function and notice that input values are in the range [0,1].
    def tangent(self, x):
        return np.tanh(x)

    # derivative for computing gradients
    def dtangent(self, x):
        return 1 - np.tanh(x) ** 2

    # lets compute a series of matrix multiplications to convert our input into our output
    def forwardProp(self):
        f = self.sigmoid(np.dot(self.f, self.x))
        self.cs *= f
        i = self.sigmoid(np.dot(self.i, self.x))
        c = self.tangent(np.dot(self.c, self.x))
        self.cs += i * c
        o = self.sigmoid(np.dot(self.o, self.x))
        self.y = o * self.tangent(self.cs)
        return self.cs, self.y, f, i, c, o

    def backProp(self, e, pcs, f, i, c, o, dfcs, dfhs):
        # error = error + hidden state derivative. clip the value between -6 and 6.
        e = np.clip(e + dfhs, -6, 6)
        # multiply error by activated cell state to compute output derivative
        do = self.tangent(self.cs) * e
        # output update = (output deriv * activated output) * input
        ou = np.dot(np.atleast_2d(do * self.dtangent(o)).T, np.atleast_2d(self.x))
        # derivative of cell state = error * output * deriv of cell state + deriv cell
        dcs = np.clip(e * o * self.dtangent(self.cs) + dfcs, -6, 6)
        # deriv of cell = deriv cell state * input
        dc = dcs * i
        # cell update = deriv cell * activated cell * input
        cu = np.dot(np.atleast_2d(dc * self.dtangent(c)).T, np.atleast_2d(self.x))
        # deriv of input = deriv cell state * cell
        di = dcs * c
        # input update = (deriv input * activated input) * input
        iu = np.dot(np.atleast_2d(di * self.dsigmoid(i)).T, np.atleast_2d(self.x))
        # deriv forget = deriv cell state * all cell states
        df = dcs * pcs
        # forget update = (deriv forget * deriv forget) * input
        fu = np.dot(np.atleast_2d(df * self.dsigmoid(f)).T, np.atleast_2d(self.x))
        # deriv cell state = deriv cell state * forget
        dpcs = dcs * f
        # deriv hidden state = (deriv cell * cell) * output + deriv output * output * output deriv input * input * output + deriv forget
        # * forget * output
        dphs = np.dot(dc, self.c)[:self.ys] + np.dot(do, self.o)[:self.ys] + np.dot(di, self.i)[:self.ys] + np.dot(df,
                                                                                                                   self.f)[
                                                                                                            :self.ys]
        # return update gradinets for forget, input, cell, output, cell state, hidden state
        return fu, iu, cu, ou, dpcs, dphs

    def update(self, fu, iu, cu, ou):
        # update forget, input, cell, and output gradients
        self.Gf = 0.9 * self.Gf + 0.1 * fu ** 2
        self.Gi = 0.9 * self.Gi + 0.1 * iu ** 2
        self.Gc = 0.9 * self.Gc + 0.1 * cu ** 2
        self.Go = 0.9 * self.Go + 0.1 * ou ** 2

        # update our gates using our gradients
        self.f -= self.lr / np.sqrt(self.Gf + 1e-8) * fu
        self.i -= self.lr / np.sqrt(self.Gi + 1e-8) * iu
        self.c -= self.lr / np.sqrt(self.Gc + 1e-8) * cu
        self.o -= self.lr / np.sqrt(self.Go + 1e-8) * ou
        return

def LoadText():
    #open text and return input and output data (series of words)
    with open("ptb.train.txt", "r") as text_file:
        data = text_file.read()
    text = list(data)
    outputSize = len(text)
    data = list(set(text))
    uniqueWords, dataSize = len(data), len(data)
    returnData = np.zeros((uniqueWords, dataSize))
    for i in range(0, dataSize):
        returnData[i][i] = 1
    returnData = np.append(returnData, np.atleast_2d(data), axis=0)
    output = np.zeros((uniqueWords, outputSize))
    for i in range(0, outputSize):
        index = np.where(np.asarray(data) == text[i])
        output[:,i] = returnData[0:-1,index[0]].astype(float).ravel()
    return returnData, uniqueWords, output, outputSize, data

#write the predicted output (series of words) to disk
def ExportText(output, data):
    finalOutput = np.zeros_like(output)
    prob = np.zeros_like(output[0])
    outputText = ""
    print(len(data))
    print(output.shape[0])
    for i in range(0, output.shape[0]):
        for j in range(0, output.shape[1]):
            prob[j] = output[i][j] / np.sum(output[i])
        outputText += np.random.choice(data, p=prob)
    with open("output.txt", "w") as text_file:
        text_file.write(outputText)
    return

#Begin program
print("Beginning")
iterations = 5000
learningRate = 0.001
#load input output data (words)
returnData, numCategories, expectedOutput, outputSize, data = LoadText()
print("Done Reading")
#init our RNN using our hyperparams and dataset
LSTM = LSTM(numCategories, numCategories, outputSize, expectedOutput, learningRate)

#training time!
for i in range(1, iterations):
    #compute predicted next word
    LSTM.forwardProp()
    #update all our weights using our error
    error = LSTM.backProp()
    #once our error/loss is small enough
    print("Error on iteration ", i, ": ", error)
    if error > -100 and error < 100 or i % 100 == 0:
        #we can finally define a seed word
        seed = np.zeros_like(RNN.x)
        maxI = np.argmax(np.random.random(RNN.x.shape))
        seed[maxI] = 1
        LSTM.x = seed
        #and predict some new text!
        output = LSTM.sample()
        print(output)
        #write it all to disk
        ExportText(output, data)
        print("Done Writing")
print("Complete")