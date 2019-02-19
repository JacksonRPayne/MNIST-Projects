import mnist
import numpy as np
import NeuralNetwork as nn
import random
import pygame
import math

# Get data from MNIST
# NOTE: I did not write the code to retrieve the data, code taken from:
# https://github.com/hsjeong5/MNIST-for-Numpy
imgTrain, lblTrain, imgTest, lblTest = mnist.load()


# Initialize input, hidden and output layers
il = np.zeros((784, 1))
hl1 = np.zeros((50,1))
hl2 = np.zeros((50,1))
middleLayer = np.zeros((2,1))
hl3 = np.zeros((50,1))
hl4 = np.zeros((50,1))
ol = np.zeros((784,1))

layers = [il, hl1,  middleLayer, hl4, ol]

# Initializes the network, loading the weights and biases from the files
# 
network = nn.NeuralNetwork(layers, learningRate=0.0003,weightImportFile="EncoderWeights.txt", biasImportFile="EncoderBiases.txt")

def vectorizeDigit(digit):
    # Initializes a 10 digit vector
    returnArray = np.zeros((10,1))
    # Sets the index of the digit to 1
    returnArray[digit] = 1
    # Returns the array
    return returnArray

def prepareInputVector(vector):
    # Copies in order to keep the original the same and to use float values
    returnVector = vector.astype(float)
    # Loops through every pixel value
    for i in range(784):
        # Normalizes the value
        returnVector[i]/=255
    # Shapes it correctly and returns it
    returnVector.shape = (784,1)
    return returnVector

def recreateImage(vector):
    # Copies to avoid changing
    returnVector = vector.copy()
    # Loops through each vector value
    for i in range(784):
        # Multiplies the value by 255, then converts it to an integer
        returnVector[i] = math.floor(returnVector[i]*255)
    
    # Returns it to its original shape, then returns it to the user
    returnVector.shape = (784,)
    return returnVector


print("Beginning training...")
'''
# Stores the size of each training batch
batchSize = 1


for i in range(100000):
    #Chooses a random training exaple index
    index = random.randint(0,59999)
    
    # Train network on data to reconstruct itself
    loss = network.computeGradientsAndDeltas(prepareInputVector(imgTrain[index]), prepareInputVector(imgTrain[index]))
    
    # After every batchsize times
    if(i%batchSize==0):
        # Apply the gradients and deltas
        network.updateWeightsAndBiases()
    
    # Print loss every hundred times
    if(i%100==0):
        # Adds up all the loss
        totalLoss = 0
        for j in loss:
            totalLoss += j**2
        # Prints loss
        print("Loss at " + str(i) + " iterations: " + str(totalLoss))


# Save the new weights and biases
network.saveWeightMatrices("EncoderWeights.txt")
network.saveBiasMatrices("EncoderBiases.txt")
'''
# Initializes the pygame window
pygame.init()

screenWidth = 400
screenHeight = 400

window = pygame.display.set_mode((screenWidth,screenHeight))
pygame.display.set_caption("MNIST Auto Encoder")

def drawData(xPos,yPos, data, size):
    # index of the data
    x = 0
    # Loop through 784 times
    for i in range(28):
        for j in range(28):
            # Draw one pixel of the digit
            pygame.draw.rect(window, (data[x],data[x],data[x]), ((j*size)+xPos,(i*size)+yPos,x,size))
            # Increase the data index
            x+=1

running = True
index=random.randint(0,9999)
network.layers[0] = prepareInputVector(imgTest[index])
# Gets the computers prediction based off the data
computerPrediction = recreateImage(network.feedForward())

network.layers[2] = np.random.random((2,1))
computerDrawing = recreateImage(network.feedForward(offset=2))

# Main loop
while running:
    # Time in between loops
    pygame.time.delay(10)
    # Loop through all events
    for i in pygame.event.get():
        # If there is a quit event
        if(i.type == pygame.QUIT):
            # Stop the loop
            running = False
        if(i.type == pygame.KEYDOWN):
            if(i.key == pygame.K_RETURN):
                # Move on to a random example
                index=random.randint(0,9999)
                # Set the input
                network.layers[0] = prepareInputVector(imgTest[index])
                # Gets the computers prediction based off the data
                computerPrediction = recreateImage(network.feedForward())
                
                network.layers[2] = np.random.random((2,1))
                computerDrawing = recreateImage(network.feedForward(offset=2))
            
    # Clear the screen
    window.fill((0,0,0))
    # Draw the test example
    drawData(0,0,imgTest[index], 5)
    # Draws the computers compressed version of it
    drawData(screenWidth-(5*28), 0, computerPrediction, 5)
    
    drawData((screenWidth/2), screenHeight-(5*28), computerDrawing, 5)
    # Update the display
    pygame.display.update()
    

print("Closing...")
# Quit the window
pygame.quit()            
