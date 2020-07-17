import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

f10 = np.loadtxt("f1/D10.txt")
f20 = np.loadtxt("f1/D20.txt")
f30 = np.loadtxt("f1/D30.txt")
f5 = np.loadtxt("f1/D5.txt")
f15 = np.loadtxt("f1/D15.txt")
f25 = np.loadtxt("f1/D25.txt")


fig = plt.figure()                    #figsize图片大小`

                     # g'green,“-”实线，label图例,一般要在名称前面加一个u
pl.plot(f5,'m-', label = u'D=5')
p2 = pl.plot(f10,'g-',label=u'D=10')
p3 = pl.plot(f15,'y-', label = u'D=15')
p4 = pl.plot(f20,'r-', label = u'D=20')
p5 = pl.plot(f25,'k-', label = u'D=25')
p6 = pl.plot(f30,'b-', label = u'D=30')

pl.legend(loc=4,bbox_to_anchor=(1, 0.2))

pl.xlabel('Epoch')
pl.ylabel('F1')
plt.title('Compare F1 for different D')
plt.show()