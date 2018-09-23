import matplotlib.pyplot as plt
import numpy as np

a = np.loadtxt('/home/doublepoints/Documents/selfdriving-T3/CarND-Semantic-Segmentation/loss/20180919222812.log')

print(a)

x = np.arange(0,50,1,dtype=int)
y =a

plt.plot(x,y)
plt.xlabel('epoch')
plt.ylabel('loss')

plt.show()
