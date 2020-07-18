import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


f5 = np.loadtxt("f1/D5.txt")
f10 = np.loadtxt("f1/D10.txt")
f15 = np.loadtxt("f1/D15.txt")
f20 = np.loadtxt("f1/D20.txt")
f25 = np.loadtxt("f1/D25.txt")
f30 = np.loadtxt("f1/D30.txt")

#fig = plt.figure()                          #括号内参数figsize=(7,5)图片大小`
pl.plot(f5,'m-', label = u'D=5')             # g'green,“-”实线，label图例,一般要在名称前面加一个u
c2 = pl.plot(f10,'g-',label=u'D=10')
c3 = pl.plot(f15,'y-', label = u'D=15')
c4 = pl.plot(f20,'r-', label = u'D=20')
c5 = pl.plot(f25,'k-', label = u'D=25')
c6 = pl.plot(f30,'b-', label = u'D=30')

pl.legend(loc=4,bbox_to_anchor=(1, 0.2))     #loc:图例位置  bbox_to_anchor：图例位置微调
pl.xlabel('Epoch')
pl.ylabel('F1')
plt.title('Compare F1 for different D')
plt.show()