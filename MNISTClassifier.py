import NeuralNetwork as nn
import numpy as np
import mnist
import pygame

# Get data from MNIST
# NOTE: I did not write the code to retrieve the data, code taken from:
# https://github.com/hsjeong5/MNIST-for-Numpy
imgTrain, lblTrain, imgTest, lblTest = mnist.load()

# Initialize input, hidden and output layers
il = np.zeros((784, 1))
hl1 = np.zeros((16,1))
hl2 = np.zeros((16,1))
ol = np.zeros((10,1))

layers = [il,hl1,hl2,ol]

# Initializes the network, loading the weights and biases from the files
# weightImportFile="weights.txt", biasImportFile="biases.txt"
network = nn.NeuralNetwork(layers, learningRate=0.1, weightImportFile="weights.txt", biasImportFile="biases.txt")
            

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

'''
print("Beginning training...")

# Training loop
for i in range(60000):
    # Train network on data with labels
    loss = network.trainStochastically(prepareInputVector(imgTrain[i]), vectorizeDigit(lblTrain[i]))
    # Print loss every thousand times
    if(i%100==0):
        # Adds up all the loss
        totalLoss = 0
        for j in loss:
            totalLoss += j**2
        # Prints loss
        print("Loss at " + str(i) + " iterations: " + str(totalLoss))
        # Gets computer guess
        guess = np.argmax(network.feedForward())
        # Gets if the guess was correct
        isCorrect = lblTrain[i] == guess
        # Prints that
        print("Got the number correct: " + str(isCorrect))
        print("Label: " + str(lblTrain[i]) + " | Guess: " + str(guess))
'''
# Holds how many correct guesses the computer has had
totalCorrect = 0
# Testing loop
for j in range(10000):
    # Sets input
    network.layers[0] = prepareInputVector(imgTest[j])
    # Gets the guess of the network
    guess = np.argmax(network.feedForward())
    # If the guess was correct
    if(lblTest[j] == guess):
        # Increment the counter
        totalCorrect +=1
        
# Calculates the percentage that were correct
computerScore = (totalCorrect/10000)*100
print("\n")
print("The percent accuracy on the testing data was: " + str(computerScore))

# Save the new weights and biases
network.saveWeightMatrices("weights.txt")
network.saveBiasMatrices("biases.txt")

print("Training done, opening pygame window...")

# Initializes the pygame window
pygame.init()

screenWidth = 400
screenHeight = 200

window = pygame.display.set_mode((screenWidth,screenHeight))
pygame.display.set_caption("MNIST Classifier")


def displayPredictionText(text):
    # Defines the text font
    font = pygame.font.Font('freesansbold.ttf',18)
    # Gets surface of text
    textSurface = font.render(str(text), True, (255,255,255))
    # Gets rect of text
    textRect = textSurface.get_rect()
    # Defines position of text
    textRect.center = (screenWidth-20,20)
    # Renders text
    window.blit(textSurface, textRect)

def drawData(data, size):
    # index of the data
    x = 0
    # Loop through 784 times
    for i in range(28):
        for j in range(28):
            # Draw one pixel of the digit
            pygame.draw.rect(window, (data[x],data[x],data[x]), (j*size,i*size,x,size))
            # Increase the data index
            x+=1

running = True
index=0

# Set the input
network.layers[0] = prepareInputVector(imgTest[index])
# Gets the computers prediction based off the data
computerPrediction = np.argmax(network.feedForward())

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
                # Move on to the next example
                index +=1
                # Set the input
                network.layers[0] = prepareInputVector(imgTest[index])
                # Gets the computers prediction based off the data
                computerPrediction = np.argmax(network.feedForward())
            
    # Clear the screen
    window.fill((0,0,0))
    # Draw the test example
    drawData(imgTest[index], 5)
    # Display the guess from the computer
    displayPredictionText(computerPrediction)
    # Update the display
    pygame.display.update()
    

print("Closing...")
# Quit the window
pygame.quit()


