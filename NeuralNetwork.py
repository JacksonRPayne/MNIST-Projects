import numpy as np
import ast

class NeuralNetwork:
    
    def __init__(self, layers, weightImportFile=None, biasImportFile=None, weightCoefficient=2, weightConstant=-1, biasCoefficient=2, biasConstant=-1, learningRate=0.1):
        # Defines the layers based on a matrix list passed
        self.layers = layers.copy()
        # The amount of weight matrices is equal to the amount of layers -1
        self.weightMatrices = [0] * (len(layers)-1)
        # The amount of bias matrices is equal to the amount of weight matrices
        self.biasMatrices = [0] * len(self.weightMatrices)
        # Used to randomize weights
        self.weightCoefficient = weightCoefficient
        self.weightConstant = weightConstant
        # Used to randomize biases
        self.biasCoefficient = biasCoefficient
        self.biasConstant = biasConstant
        # Initializes learning rate (defaults to 0.1)
        self.learningRate = learningRate
        # Initializes the list of gradients as empty
        self.gradients = [0] * len(self.biasMatrices)
        # Initializes the deltas of the weights as empty
        self.weightDeltas = [0] * len(self.weightMatrices)
        # Initializes weight and bias matrices lists
        for i in range(len(self.weightMatrices)):
            # Initializes a weight matrix with the rows equal to the number of nodes in this layer 
            # and the columns equal to the number of nodes in the next layer
            weightMatrix = self.weightCoefficient*np.random.random((layers[i+1].size, layers[i].size))+self.weightConstant
            # Puts this matrix into the matrix list at index i
            self.weightMatrices[i] = weightMatrix
            # Creates bias matrix (1 column) with the same size as the next layer
            biasMatrix = self.biasCoefficient*np.random.random((layers[i+1].size, 1))+self.biasConstant
            # Inserts this bias matrix into the matrices list at the index i
            self.biasMatrices[i] = biasMatrix
            
        # Creating a list to store the shapes of each weight matrix
        self.weightMatrixShapes = [0] * len(self.weightMatrices)
        # Iterates through each weight matrix
        for i in range(len(self.weightMatrices)):
            # Stores the shape of the matrix in the list
            self.weightMatrixShapes[i] = self.weightMatrices[i].shape
            
        # Creating a list to store the shapes of each bias matrix
        self.biasMatrixShapes = [0] * len(self.biasMatrices)
        # Iterates through each bias matrix
        for i in range(len(self.biasMatrices)):
            # Stores the shape of the matrix in the list
            self.biasMatrixShapes[i] = self.biasMatrices[i].shape
        
        # Initializes the array sizes of the gradient and deltas lists
        self.resetGradientsAndDeltas()
        
        # If there is a specified file to import weights    
        if(weightImportFile != None):
            # Load weights from this file
            self.loadWeightMatrices(weightImportFile)
        # If there is a specified file to import biases
        if(biasImportFile !=None):
            # Load biases from that file
            self.loadBiasMatrices(biasImportFile)
    
    def feedForward(self):
        for i in range(len(self.layers)-1):  
            # Initializes the layer as the dot product of the weight matrix and the last layer
            self.layers[i+1] = np.dot(self.weightMatrices[i], self.layers[i])
            # Adds a bias to each neuron
            self.layers[i+1] += self.biasMatrices[i]
            # Runs the whole layer through a sigmoid
            self.layers[i+1] = NeuralNetwork.sigmoid(self.layers[i+1])
        
        # Returns the output layer    
        return self.layers[len(self.layers)-1]
            
    
    def trainStochastically(self, inputs, correctOutputs):
        # Sets the input layer
        self.layers[0] = inputs
        # Feeds forward the network
        outputs = self.feedForward()
        # If the parameter isn't the same size as the output layer
        if(correctOutputs.size != outputs.size):
            # Send error and return out of function
            print("Wrong dimensioned outputs in trainStochastically function")
            return
        # Gets the error of the output layer
        outputError = correctOutputs - outputs
        # A list that stores the error values going backwards (first the output error, then hidden 1, etc.)
        errors = [outputError]
        # Loops through the hidden layers
        for i in range(len(self.layers)-2):
            # Adds the error of each hidden layer, working backwards by multiplying
            # the transpose of the weight matrix by the error in the forward layer
            errors.append(np.dot(self.weightMatrices[len(self.weightMatrices)-(i+1)].T, errors[i]))
        
        for j in range(len(self.weightMatrices)):
            # Gets the gradient of a weight matrix in relation to the error of the forward layer
            gradient = self.learningRate * (NeuralNetwork.dSigmoid(self.layers[len(self.layers)-(j+1)]) * errors[j])
            # Updates the bias matrix according to the gradient
            self.biasMatrices[len(self.biasMatrices)-(j+1)] = self.biasMatrices[len(self.biasMatrices)-(j+1)] + gradient
            # Gets the delta of the weight matrix by multiplying the gradient with the transpose of the previous layer
            delta = np.dot(gradient, self.layers[len(self.layers)-(j+2)].T)
            # Updates the weights according to the delta
            self.weightMatrices[len(self.weightMatrices)-(j+1)] = self.weightMatrices[len(self.weightMatrices)-(j+1)]+delta
         
        # Returns the error of the output layer 
        return outputError
    
    
    def computeGradientsAndDeltas(self, inputs, correctOutputs):
        # Sets the input layer
        self.layers[0] = inputs
        # Feeds forward the network
        outputs = self.feedForward()
        # If the parameter isn't the same size as the output layer
        if(correctOutputs.size != outputs.size):
            # Send error and return out of function
            print("Wrong dimensioned outputs in computeGradientsAndDeltas function")
            return
        # Gets the error of the output layer
        outputError = correctOutputs - outputs
        # A list that stores the error values going backwards (first the output error, then hidden 1, etc.)
        errors = [outputError]
        # Loops through the hidden layers
        for i in range(len(self.layers)-2):
            # Adds the error of each hidden layer, working backwards by multiplying
            # the transpose of the weight matrix by the error in the forward layer
            errors.append(np.dot(self.weightMatrices[len(self.weightMatrices)-(i+1)].T, errors[i]))
        
        for j in range(len(self.weightMatrices)):
            # Gets the gradient of a layer in relation to the error of the forward layer
            gradient = self.learningRate * (NeuralNetwork.dSigmoid(self.layers[len(self.layers)-(j+1)]) * errors[j])
            # Updates the bias matrix according to the gradient
            self.gradients[len(self.gradients)-(j+1)] = self.gradients[len(self.gradients)-(j+1)] + gradient
            # Gets the delta of the weight matrix by multiplying the gradient with the transpose of the previous layer
            delta = np.dot(gradient, self.layers[len(self.layers)-(j+2)].T)
            # Updates the weights according to the delta
            self.weightDeltas[len(self.weightDeltas)-(j+1)] = self.weightDeltas[len(self.weightDeltas)-(j+1)]+delta
         
        # Returns the error of the output layer 
        return outputError
    
    def resetGradientsAndDeltas(self):
        # Resets arrays to empty
        self.gradients = [0] * len(self.biasMatrices)
        self.weightDeltas = [0] * len(self.weightMatrices)
        
        # Loop through all weight matrices
        for i in range(len(self.weightMatrices)):
            # Sets each delta value to an array of zeros with the same shape as each weight matrix
            self.weightDeltas[i] = np.zeros(self.weightMatrices[i].shape)
        
        # Loop through all bias matrices
        for j in range(len(self.biasMatrices)):
            # Sets each gradient value to an array of zeros with the same shape as each bias vector
            self.gradients[j] = np.zeros(self.biasMatrices[j].shape)
    
    def updateWeightsAndBiases(self):
        # Loop through all weight matrices
        for i in range(len(self.weightMatrices)):
            # Adds the delta of that weight matrix to that weight matrix
            self.weightMatrices[i] = self.weightMatrices[i] + self.weightDeltas[i]
        
        # Loop through all bias matrices
        for j in range(len(self.biasMatrices)):
            # Adds each gradient to each bias vector
            self.biasMatrices[j] = self.biasMatrices[j] + self.gradients[j]
            
        # Resets the values of the gradients and deltas
        self.resetGradientsAndDeltas()
        
    
    # Returns sigmoid of x
    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    # Returns "derivative" of sigmoid of x
    @staticmethod
    def dSigmoid(x):
        return x*(1-x)
    
    def saveMatrixList(self, fileName, matrixList):
        # Creates a copy of the matrix list to avoid altering it
        matrixListCopy = matrixList.copy()
        # Converts every value in the copied list to a string
        for i in range(len(matrixListCopy)):
            matrixListCopy[i] = matrixListCopy[i].tostring()
        # Saves a string representation of the list
        stringList = str(matrixListCopy)        
        # Opens file with specified file name, and specifies that we're writing to it
        saveFile = open(fileName, "w")
        # Writes to the file
        saveFile.write(stringList)
        # Closes the file
        saveFile.close()

        
    def loadMatrixList(self, fileName, matrixList):
        # Opens the file, specifying we want to read
        readFile = open(fileName, "r")
        # Constructs a list from the string in the file
        newMatrixList = ast.literal_eval(readFile.read())
        # Iterates through each value of the constructed list
        for i in range(len(newMatrixList)):
            # Converts each string back into an array
            newMatrixList[i] = np.fromstring(newMatrixList[i])
        
        # Returns a copy of the constructed list
        return newMatrixList.copy()
    
    def saveWeightMatrices(self, fileName):
        self.saveMatrixList(fileName, self.weightMatrices)
    
    def loadWeightMatrices(self, fileName):
        # Loads the weight matrix list from a file
        newWeightList = self.loadMatrixList(fileName, self.weightMatrices)
        # Iterates through the loaded list
        for i in range(len(newWeightList)):
            # Sets the shape of the matrix to what it was before converting to a string
            newWeightList[i].shape = self.weightMatrixShapes[i]
            
        # Sets the weightMatrices to the list constructed from the file
        self.weightMatrices = newWeightList.copy()
    
    def saveBiasMatrices(self, fileName):
        self.saveMatrixList(fileName, self.biasMatrices)
    
    def loadBiasMatrices(self, fileName):
        # Loads the weight matrix list from a file
        newBiasList = self.loadMatrixList(fileName, self.biasMatrices)
        # Iterates through the loaded list
        for i in range(len(newBiasList)):
            # Sets the shape of the matrix to what it was before converting to a string
            newBiasList[i].shape = self.biasMatrixShapes[i]
            
        # Sets the weightMatrices to the list constructed from the file
        self.biasMatrices = newBiasList.copy()
    
