from kmeans import Kmeans
import pandas as pd
import sys
#import data and arguments
kstring = sys.argv[1]
k = int(kstring) #takes the kstring and makes it a int value
data = pd.read_csv(sys.argv[2], delim_whitespace=True, names=('x', 'y')).values
x = Kmeans(data,k)
x.cluster()
x.printData()
