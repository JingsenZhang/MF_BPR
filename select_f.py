import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

f10 = np.loadtxt("rmse/rmse10.txt")
f20 = np.loadtxt("rmse/rmse20.txt")
f30 = np.loadtxt("rmse/rmse30.txt")
f5 = np.loadtxt("rmse/rmse5.txt")
f15 = np.loadtxt("rmse/rmse15.txt")
f25 = np.loadtxt("rmse/rmse25.txt")


fig = plt.figure(figsize = (7,5))                    #figsize图片大小`

pl.plot(f10,'g-',label=u'F=10')                       # g'green,“-”实线，label图例,一般要在名称前面加一个u
p2 = pl.plot(f20,'r-', label = u'F=20')
p3 = pl.plot(f30,'b-', label = u'F=30')
p4 = pl.plot(f5,'m-', label = u'F=5')
p5 = pl.plot(f15,'y-', label = u'F=15')
p6 = pl.plot(f25,'k-', label = u'F=25')
pl.legend()

pl.xlabel('Epoch')
pl.ylabel('RMSE')
plt.title('Compare RMSE for different F')
plt.show()