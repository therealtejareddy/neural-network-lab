### EX NO : 06
### DATE  : 29.04.2022
# <p align="center"> ANN BY BACK PROPAGATION ALGORITHM </p>
## Aim:
   To implement multi layer artificial neural network using back propagation algorithm.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner /Google Colab

## Related Theory Concept:
Algorithm for ANN Backpropagation: 
• Weight initialization: 
        Set all weights and node thresholds to small random numbers. Note that the node threshold is the negative of the weight from the bias unit(whose activation level is fixed at 1). 
 
• Calculation of Activation: 
</br>
1.	The activation level of an input is determined by the instance presented to the network. 
2.	The activation level oj of a hidden and output unit is determined. 

• Weight training:

1.	Start at the output units and work backward to the hidden layer recursively and adjust weights. 

2.	The weight change is completed. 

3.	The error gradient is given by: 

a.	For the output units. 

b.	For the hidden units.

4.	Repeat iterations until convergence in term of the selected error criterion. An iteration includes presenting an instance, calculating activation and modifying weights. 

## Algorithm
1.Import packages

2.Defining Sigmoid Function for output

3.Derivative of Sigmoid Function

4.Initialize variables for training iterations and learning rate

5.Defining weight and biases for hidden and output layer

6.Updating Weights

## Program:
```python
"""
Program to implement ANN by back propagation algorithm.
Developed by: MADITHATI YUVATEJA REDDY
Register Number: 212219040069
"""
import numpy as np
X=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
X=X/np.amax(X,axis=0)
y=y/100

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivatives_sigmoid(x):
    return x*(1-x)

epoch=7000
lr=0.1
inputlayer_neuron=2
hiddenlayer_neuron=3
output_neuron=1

wh=np.random.uniform(size=(inputlayer_neuron,hiddenlayer_neuron))
bh=np.random.uniform(size=(1,hiddenlayer_neuron))
wout=np.random.uniform(size=(hiddenlayer_neuron,output_neuron))
bout=np.random.uniform(size=(1,output_neuron))

for i in range(epoch):
    hinp1=np.dot(X,wh)
hinp=hinp1+bh
hlayer_act=sigmoid(hinp)
outinp1=np.dot(hlayer_act,wout)
outinp=outinp1+bout
output=sigmoid(outinp)

EO=y-output
outgrad=derivatives_sigmoid(output)
d_output=EO* outgrad
EH=d_output.dot(wout.T)
hiddengrad=derivatives_sigmoid(hlayer_act)
d_hiddenlayer=EH*hiddengrad
wout+=hlayer_act.T.dot(d_output)*lr
wh+=X.T.dot(d_hiddenlayer)*lr
print("Input: \n"+str(X))
print("Actual Output: \n"+str(y))
print("Predicted Output: \n",output)
```

## Output:
![image](https://user-images.githubusercontent.com/75235293/169482242-884d3f02-5874-4606-9f99-84c5611adbb0.png)



## Result:
Thus the python program successully implemented multi layer artificial neural network using back propagation algorithm.