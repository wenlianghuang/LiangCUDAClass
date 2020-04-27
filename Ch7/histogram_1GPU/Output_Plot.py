import numpy as np
import matplotlib.pyplot as plt 
from scipy import optimize
import sys
datfile = sys.argv[1] #Set the parametr 1 to .dat file name

#Fit function with .dat file data
def test_func(x,a,b):
    return a * np.exp(-b*x)

hist_data = [] 
y_data = []
with open(datfile,'r') as f:
    d = f.readlines()

    for i in d:
        histvalue = i.split(" ")[0]
        yvalue = i.split(" ")[1]
        histvalue = float(histvalue)
        yvalue = float(yvalue)
        hist_data.append(histvalue)
        y_data.append(yvalue)
hist_data = np.array(hist_data)
y_data = np.array(y_data)

#parameters with curve fit
params, params_covariance = optimize.curve_fit(test_func, hist_data, y_data)
fig = plt.figure()
plt.title(datfile.split(".")[0] + " Output")
x = np.linspace(0,20,101)
y = test_func(hist_data,*params)
plt.bar(hist_data,y_data,color='#99ff33',label='Data')
plt.plot(x,y,'r',label='fit: a=%5.3f, b=%5.3f' %(params[0],params[1]))
plt.xlabel('hist_value')
plt.ylabel('y_value')

plt.legend()
plt.show()
if(datfile.find("gmem") != -1):
    fig.savefig("Output_gmem.png")
elif(datfile.find("shmem") != -1):
    fig.savefig("Output_sheme.png")
else:
    fig.savefig("Output_cpu.png")
plt.close()

