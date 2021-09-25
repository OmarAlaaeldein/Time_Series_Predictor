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
        plt.plot(self.x,np.cumsum(self.data),label="True Values")
        plt.legend()
        plt.show()
        self.cofs=a
    def get_coefficients(self):
        """returns a printable form of the the polynomial coefficients in the LSF model
        """
        for i,j in enumerate(list(self.cofs)):
            print('a{}= {}'.format(i,j))
a=TimeSeries(r"cases.npy")
a.model_fit(3) # modify this as you see fit
a.get_coefficients() 