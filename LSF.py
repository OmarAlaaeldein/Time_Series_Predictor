import numpy as np
from numpy.linalg import inv, solve
import matplotlib.pyplot as plt
f=np.load("cases.npy")
x=np.array(np.arange(0,len(f),1))
r=[]
c,t=[],0
order=2 # modify this as you see fit
plt.axvline(x=0,color='k'),plt.axhline(y=0,color='k'),plt.grid()
#Least Square Fitting Model
A=np.zeros(shape=(len(x),order+1))
print(A)
for i in range(order+1):
    A[:,i]=x**i #A[start row:end row,start column:end column]
print(A)
a=solve(A.T.dot(A),A.T.dot(f)) #find factors of the polynomial using LSF
for i in range(len(x)):
    d=A[i,:].dot(a)
    c.append(d)
    plt.scatter(i,sum(c),color='red')
plt.plot(x,np.cumsum(f),label="True Values")
plt.legend()
for i,j in enumerate(list(a)):
    print('a{}={}'.format(i,j))
plt.show()