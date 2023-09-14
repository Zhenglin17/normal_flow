# import matplotlib.pyplot as plt

# xstrings = ['Q2W1','Q2W2','Q2W3','Q2W4','Q2W5','Q2W6','Q2W7','Q2W8','Q2W9','Q2W10','Q2QW11','Q2W12','Q2W13']
# y = [88,84,83,99,96,85,85,82,65,60,19,45,27]
# xs = []
# ys = []

# plt.plot(xstrings, y, linestyle='-', marker='.', color='#009d9a', linewidth=1)
# #plt.legend(['data', 'linear', 'cubic'], loc='best')

# for a,b in zip(xstrings,y):
#    xs.append(a)
#    ys.append(b)
#    if b < 7:
#        label = "${:,.2f}".format(b)
#    elif b < 10:
#        label = "${:,.1f}".format(b)
#    else:
#        label = "${:,.0f}".format(b)
#    plt.annotate(label, # this is the text
#                 (a,b), # this is the point to label
#                 textcoords="offset points", # how to position the text
#                 xytext=(0,3), # distance from text to points (x,y)
#                 ha='center', fontsize = 6)


# plt.xticks(fontsize=6.5, rotation=45)
# plt.show()
# #plt.savefig("Graphs/"+'test.png', dpi=300, bbox_inches='tight')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

Y = np.array([12, 7, 6, 9, 11, 10, 13, 17, 15, 16])
x = np.array([1, 1.5, 2, 4, 4.5, 6, 7, 9, 9.5, 10])
f = interp1d(x, Y)
f2 = interp1d(x, Y, kind = 'cubic')

xnew = np.linspace(1, 10, num=100, endpoint=True)
plt.plot(x, Y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')

plt.show()