import numpy as np
from numpy.linalg import inv, solve
import matplotlib.pyplot as plt
class TimeSeries:

    def __init__(self,path: str):
        self.data=np.load(path) #  a pickled 1D array will work too
        self.x=np.array(np.arange(0,len(self.data),1))
        self.c,self.t=[],0
        plt.figure()
        plt.axvline(x=0,color='k'),plt.axhline(y=0,color='k'),plt.grid()
        self.cofs=[]
        self.predicted_pairs=np.zeros(len(self.x))

    def model_fit(self,order: int):     #Least Square Fitting Model
        """Perform LSF for a given array with the corresponding order of regrsssion

        Args:
            order (int): increases how the polynomial acts in accordance with data
        """
        A=np.zeros(shape=(len(self.x),order+1))
        for i in range(order+1):
            A[:,i]=self.x**i #A[start row:end row,start column:end column]
        a=solve(A.T.dot(A),A.T.dot(self.data)) #find factors of the polynomial equation using LSF
        for i in range(len(self.x)):
            d=A[i,:].dot(a)
            self.c.append(d)
            plt.scatter(i,sum(self.c),color='red')
            self.predicted_pairs[i]=d
        plt.plot(self.x,np.cumsum(self.data),label="True Values")
        plt.legend()
        plt.show()
        self.cofs=a
    def get_coefficients(self):
        """returns a printable form of the the polynomial coefficients in the LSF model
        """
        for i,j in enumerate(list(self.cofs)):
            print('a{}= {}'.format(i,j))
        return self.cofs
    def R2(self,):
        avg=np.mean(self.data)
        sst=np.sum((np.cumsum(self.data-avg))**2)
        sse=np.sum((np.cumsum(self.data)-np.cumsum(self.predicted_pairs))**2)
        print("R2={}".format((sst-sse)/sst))
        return (sst-sse)/sst

a=TimeSeries(r"cases.npy")
a.model_fit(1) # modify this as you see fit, actully setting this with 40 aka len(self.x) will produce a perfect R2 score since it is an interpolation of the whole dataset
a.get_coefficients()
a.R2()