import numpy as np
from copy import deepcopy
class Kmeans:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        #creating random centroids
        self.centroids = data[np.random.randint(data.shape[0], size=k)]
        #store old centroids
        self.centroidsPrev = np.zeros(self.centroids.shape)
        #create labels with zeros to shape of data
        self.labels = np.zeros(data.shape)
        #distance between centroids
        self.distances = np.zeros([data.shape[0], k])
#Cluster takes the data and the centroids and gets the labels
#then calls updateCentroids to update the centroids to the new centroids
    def cluster(self):
        #while centroidsPrev - centroids is not equal to 0 it goes thorw the loop
        #when it does equal zero it stops
        while(self.centroidsPrev - self.centroids).all() != 0:
            for i in range(self.k):
                for j, c in enumerate(self.centroids):
                    self.distances[:, j] = euclidean_distance(c,self.data) # gets the Euclidean distance between the points
                self.labels = np.argmin(self.distances, axis=1) # sets the labels
                self.centroidsPrev = deepcopy(self.centroids) #copies old centroids to new centroids
                updateCentroids(self)
#printData prints the data with the labels
    def printData(self):
        #adds 1 to the labels so that they are 0=1, 1=2 so on
        for i in range(len(self.data)):
            self.labels[i] = self.labels[i] + 1
            #printing to file
            f= open("output.txt","w")
            for i in range(len(self.data)):
                f.write( str(self.data[i]) + "\t " + str(self.labels[i]) + "\n")
            f.close()
#updates the centroids
def updateCentroids(self):
    for i in range(self.k):
        self.centroids[i] = np.average(self.data[self.labels == i], 0)
#fuction for euclidean distance using numpy
def euclidean_distance(a,b):
    return np.linalg.norm(a-b, axis=1)
