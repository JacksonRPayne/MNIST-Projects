import mnist
import numpy as np

imgTrain, lblTrain, imgTest, lblTest = mnist.load()

def prepareInputVector(vector):
    # Copies in order to keep the original the same
    returnVector = vector.astype(float)
    # Loops through every pixel value
    for i in range(784):
        if(returnVector[i]==255):
            print("WOWEE")
        # Normalizes the value
        returnVector[i]= returnVector[i]/255
    
    # Shapes it correctly and returns it
    returnVector.shape = (784,1)
    return returnVector


print(imgTrain[0])
print(prepareInputVector(imgTrain[0]))