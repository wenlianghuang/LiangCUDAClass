import numpy as np
import matplotlib.pyplot as plt

data = [] #Prepare to grab the phi_GPU.dat into list "data"
with open('phi_GPU.dat','r') as f:
    d = f.readlines() #All lines in the phi_GPU.dat
    for i in d:
        k = i.split(" ") 
        data.append([i for i in k])

data = np.array(data)
data_fixed = np.delete(data,np.s_[-1],1) #delete all the last array of the column
data_fixed = data_fixed.astype(np.float) #convert string to float
plt.imshow(data_fixed)
plt.colorbar()
plt.show()
