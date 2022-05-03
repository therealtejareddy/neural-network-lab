"""
Program to implement the multi class classifier.
Developed by: MADITHATI YUVATEJA REDDY
RegisterNumber: 212219040069
"""
#BACK PROPOGATION IN SINGLE NEURON
import numpy as np
i=1.5
w_o=0.0
y=0.5
r=0.01
def dc_dw(a,y,i):
    dc_da=2*(a-y)
    da_dw=i
    return dc_da*da_dw
w=[w_o]
a=[w_o*i]
for j in range(0,100):
    a.append(w[-1]*i)
    w.append(w[-1]-r*dc_dw(a[-1],y,i))
    if(a[-1]-y)**2<0.001:
        break
print(a)
print(" ")
print(w)