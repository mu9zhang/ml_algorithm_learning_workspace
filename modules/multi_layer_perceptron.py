import numpy as np

class mlp:
    """ A Multi-Layer Peceptron """
    
    def __init__(self, inputs, targets, nhidden, beta=1, momentum=0.9, outtype='logistic'):
       """ Constructor """
       # Set up the network size
       self.nin   = np.shape(inputs)[1]
       self.nout  = np.shape(targets)[1]
       self.ndata = np.shape(inputs)[0]
       self.nhidden = nhidden

       # Hyperparameter
       self.beta = beta
       self.momentum = momentum
       self.outtype = outtype

       # Initialize network: (-1/sqrt(n), 1/sqrt(n)), where n is the input
       self.weights1 = ((np.random.rand(self.nin + 1, self.nhidden) - 0.5) * 2 / np.sqrt(self.nin))
       self.weights2 = ((np.random.rand(self.nhidden + 1, self.nout) - 0.5) * 2 / np.sqrt(self.nhidden))

    def earlystopping(self, inputs, targets, valid, validtargets, eta, niteration=100):
        """ Early Stopping """
        # Add the valid that match the bias node
        valid = np.concatenate((valid, -np.ones((np.shape(valid)[0], 1))), axis=1)
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000

        count = 0
        while ((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1) > 0.001):
            count = count + 1
            print(count)
            self.mlptrain(inputs, targets, eta, niteration)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5 * np.sum((validtargets - validout) ** 2)

        print("Stopped: ", new_val_error, old_val_error1, old_val_error2)
        return new_val_error

    def mlptrain(self, inputs, targets, eta, niteration):
        """ Train the thing """
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs, -np.ones((self.ndata, 1))), axis=1)
        change = range(self.ndata)

        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))

        for n in range(niteration):
            self.outputs = self.mlpfwd(inputs)

            error = 0.5 * np.sum((self.outputs - targets) ** 2)
            if (n % 100 == 0):
                print("Iteration: ", n, "\tError: ", error)

            # Backprop from output -> hidden
            # Different types of output neurons
            if self.outtype == 'linear':
                deltao = (self.outputs - targets) / self.ndata
            elif self.outtype == 'logistic':
                deltao = self.beta * (self.outputs - targets) * self.outputs * (1- self.outputs)
            elif self.outtype == 'softmax':
                deltao = (self.outputs - targets) * (self.outputs * (-self.outputs) + self.outputs) / self.ndata 
            else:
                print("error")

            # Backprop from hidden -> input
            deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2)))

            # Update weights
            updatew1 = eta * (np.dot(np.transpose(inputs), deltah[:,:-1])) + self.momentum * updatew1
            updatew2 = eta * (np.dot(np.transpose(self.hidden), deltao)) + self.momentum * updatew2
            self.weights1 -= updatew1
            self.weights2 -= updatew2

            # Randomise order of inputs (not necessary for matrix-based calculation)
            #np.random.shuffle(change)
            #inputs = inputs[change,:]
            #targets = targets[change,:]

    def mlpfwd(self, inputs):
        """ Run the network forward  """
        # Input -> Hidden
        self.hidden = np.dot(inputs, self.weights1)
        self.hidden = 1.0 / (1.0 + np.exp(-self.beta * self.hidden))
        self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)

        # Hidden -> Output
        outputs = np.dot(self.hidden, self.weights2)

        # Different types of output neurons
        if self.outtype == 'linear':
            # regression task
            return outputs
        elif self.outtype == 'logistic':
            # 0/1 classification task
            return 1.0 / (1.0 + np.exp(-self.beta * outputs))
        elif self.outtype == 'softmax':
            # multi-class classification task
            normalizer = np.sum(np.exp(outputs), axis=1) * np.ones((1, np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs)) / normalizer)
        else:
            print("error")

    def confmat(self, inputs, targets):
        """ Confusion matrix """
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.mlpfwd(inputs)

        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs>0.5, 1, 0)
        else:
            # 1-of-N encoding
           outputs = np.argmax(outputs, 1)
           targets = np.argmax(targets, 1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))
        
        print("Confusion matrix is:")
        print(cm)
        print("Percentage Correct: ", np.trace(cm)/np.sum(cm)*100)
