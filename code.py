#Intro to QML: Building a Classifier based on PQC:

#Prerequisties
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


#Demonstrations:

#1. XOR Gate Output Prediction using PQC:

# Using XOR Gate Prediction as the first QML demo using PQC.
# As, XOR is a Non Linear Gate, Thus the prediction pattern is also Non Linear. Hence, being used for simplicity of the first demo.
# For XOR, (0,0) , (1,1) are 0 and (0,1) , (1,0) are 1. Hence, Encoding those Rules as Data.

# I. Data for Model / Rules of XOR in Numerical Form

#i. Inputs for XOR
X1 = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

#ii. Outputs for XOR
Y1 = np.array([0, 1, 1, 0])

#iii. Input as x1
#Defined as x1 = X1[i] using for loop in the argument of function prob()

#iv. Initial Parameters (Theta)
theta = np.random.uniform(0, 2 * np.pi, 6) #Randomly initializing 6 parameters for 3 iterations


# II. Defining Circuit as Function (as it can only run for 1 input of X1 (for 2 gates as 2 qubits) at a time, this should be called 4 times)

def circuit_initialization(x1, theta):

    #x is the input data (X1[i])
    #theta is the training parameters (theta[i])

    qc = QuantumCircuit(2) #As there are 2 features (X1[0][0] and X1[0][1] (For first element, 0, 0 as first and second input respectively)), we need 2 qubits (as one qubit holds one features/gates)
   
    #Defining conversion of classical data to quantum states

    #Syntax for encoding data: qc.gate(operation, qubit)

    #Input Data to Rotation in Space
    qc.ry(x1[0] * np.pi, 0) #Encoding the state of gate 1 as [0](first input)
    qc.ry(x1[1] * np.pi, 1) #Encoding the state of gate 2 as [1](second input)

    #Step 1: Training Parameters (accessing the first and second element of theta and rotating accordingly)
    qc.ry(theta[0], 0) #First training parameter
    qc.ry(theta[1], 1) #Second training parameter

    #Step 2: Entangling 2 qubits
    qc.cx(0, 1) 

    #Step 3: Training Parameters (accessing the third and fourth element of theta and rotating accordingly)
    qc.ry(theta[2], 0) #Third training parameter
    qc.ry(theta[3], 1) #Fourth training parameter

    #Step 4: Re - Entangling 2 qubits
    qc.cx(0, 1)

    #Step 5: Training Parameters (accessing the fifth and sixth element of theta and rotating accordingly)
    qc.ry(theta[4], 0) #Fifth training parameter
    qc.ry(theta[5], 1) #Sixth training parameter

    #Through this, 3 iterations of Rotation would be done, before displaying the output.

    return qc


# III. Finding Probability for Each Input (using as function as it will be iterated for each input)

def prob(x1, theta):

    #i. Getting Circuit for Inputs (based off the value of i in for loop)
    qc = circuit_initialization(x1, theta)

    #ii. Getting Outputs as Statevector
    sv = Statevector.from_instruction(qc) #from_instruction is qiskit function

    #iii. Getting Probabilities from Statevector
    probs = sv.probabilities()

    #iv. Getting Probability of output (y) of 1
    prob = probs[1] + probs[2] #probs[0] is 0,0, probs[1] is 0,1, probs[2] is 1,0, probs[3] is 1,1

    #v. Returning Probability
    return prob


# IV. Computing Loss (MSE) (as a function as it would be iterated as a function for each input)

def loss(theta, X1, Y1):
    
    #i. Initializing total loss as 0
    total = 0 

    #ii. Iterating for each input
    for i in range(len(X1)):

        #Getting probability for each input (calling prob fxn, passing X1[i] and random theta, getting prob for y = 1 for that input as p)
        p = prob(X1[i], theta) 

        #Calculating loss
        total += (p - Y1[i])**2  # Predicted - Real (sq)

    #iii. Returning Avg Loss
    avgloss = (total) / (len(X1)) 
    return avgloss


# V. Training (Using Gradient Descent)
    
#i. Initializing epsilon for GD
eps = 0.05

print("\n")

print(f"Starting Training 1\n")

print(f"Initial Loss: {loss(theta, X1, Y1):.4f} \n")

#ii. Initializing LR and Epochs
epochs = 150
lr = 1.0000

#iii. Gradient Descent Algorithm
for epoch in range(epochs):
    grad = np.zeros_like(theta) #Initializing Gradient as 0
        
    # Calculating numerical gradient for each parameter
    for i in range(len(theta)):
        thetaplus = theta.copy()
        thetaplus[i] += eps
        lossplus = loss(thetaplus, X1, Y1)
            
        thetaminus = theta.copy()
        thetaminus[i] -= eps
        lossminus = loss(thetaminus, X1, Y1)
            
        grad[i] = (lossplus - lossminus) / (2 * eps)
            
    #Update parameters (Gradient Descent step)
    theta -= lr * grad
        
    # Log progress
    if (epoch + 1) % 20 == 0 or epoch == 0:
        currentloss = loss(theta, X1, Y1)
        print(f"Epoch {epoch+1:3d}/{epochs} & Loss: {currentloss:.4f}")

# VI. Results
print("\n")
print("Final Results (for XOR Classification)")
print("\n")
    
finalhits = 0
for i in range(len(X1)):
    p = prob(X1[i], theta)
    pred = 1 if p > 0.5 else 0
    if pred == Y1[i]: finalhits += 1

    print(f"Input: {X1[i]} , Prob(1): {p:.4f}")
    print(f"Pred: {pred} , Actual: {Y1[i]}")
    print("\n")

print(f"Model Accuracy: {finalhits / len(X1) * 100:.4f}%")



#2. 2D Point Output Prediction using PQC:

# Using 2D Point Prediction as the SECOND QML demo using PQC.
# As, 2D Point Prediction is Non Linear (as a set of non linear points are chosen), Thus the prediction pattern is also Non Linear. Hence, being used for simplicity of the second demo.
# For 2D Point Prediction, (0,0) , (1,1) are 0 and (0,1) , (1,0) are 1. Hence, Encoding those Rules as Data.

# I. Data for Model / Rules of 2D Point Prediction in Numerical Form

#i. Inputs for 2D Point Prediction
X2 = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

#ii. Outputs for 2D Point Prediction
Y2 = np.array([0, 1, 1, 0])

#iii. Input as x2
#Defined as x2 = X2[i] using for loop in the argument of function prob()

#iv. Initial Parameters (Theta)
theta = np.random.uniform(0, 2 * np.pi, 6) #Randomly initializing 6 parameters for 3 iterations


# II. Defining Circuit as Function (as it can only run for 1 input of X1 (for 2 pts as 2 qubits) at a time, this should be called 4 times)

def circuit_initialization(x2, theta):

    #x is the input data (X1[i])
    #theta is the training parameters (theta[i])

    qc = QuantumCircuit(2) #As there are 2 features (X1[0][0] and X1[0][1] (For first element, 0, 0 as first and second input respectively)), we need 2 qubits (as one qubit holds one features/pts)
   
    #Defining conversion of classical data to quantum states

    #Syntax for encoding data: qc.gate(operation, qubit)

    #Input Data to Rotation in Space
    qc.ry(x2[0] * np.pi, 0) #Encoding the state of gate 1 as [0](first input)
    qc.ry(x2[1] * np.pi, 1) #Encoding the state of gate 2 as [1](second input)

    #Step 1: Training Parameters (accessing the first and second element of theta and rotating accordingly)
    qc.ry(theta[0], 0) #First training parameter
    qc.ry(theta[1], 1) #Second training parameter

    #Step 2: Entangling 2 qubits
    qc.cx(0, 1) 

    #Step 3: Training Parameters (accessing the third and fourth element of theta and rotating accordingly)
    qc.ry(theta[2], 0) #Third training parameter
    qc.ry(theta[3], 1) #Fourth training parameter

    #Step 4: Re - Entangling 2 qubits
    qc.cx(0, 1)

    #Step 5: Training Parameters (accessing the fifth and sixth element of theta and rotating accordingly)
    qc.ry(theta[4], 0) #Fifth training parameter
    qc.ry(theta[5], 1) #Sixth training parameter

    #Through this, 3 iterations of Rotation would be done, before displaying the output.

    return qc


# III. Finding Probability for Each Input (using as function as it will be iterated for each input)

def prob(x2, theta):

    #i. Getting Circuit for Inputs (based off the value of i in for loop)
    qc = circuit_initialization(x2, theta)

    #ii. Getting Outputs as Statevector
    sv = Statevector.from_instruction(qc) #from_instruction is qiskit function

    #iii. Getting Probabilities from Statevector
    probs = sv.probabilities()

    #iv. Getting Probability of output (y) of 1
    prob = probs[1] + probs[2] #probs[0] is 0,0, probs[1] is 0,1, probs[2] is 1,0, probs[3] is 1,1

    #v. Returning Probability
    return prob


# IV. Computing Loss (MSE) (as a function as it would be iterated as a function for each input)

def loss(theta, X2, Y2):
    
    #i. Initializing total loss as 0
    total = 0 

    #ii. Iterating for each input
    for i in range(len(X2)):

        #Getting probability for each input (calling prob fxn, passing X1[i] and random theta, getting prob for y = 1 for that input as p)
        p = prob(X2[i], theta) 

        #Calculating loss
        total += (p - Y2[i])**2  # Predicted - Real (sq)

    #iii. Returning Avg Loss
    avgloss = (total) / (len(X2)) 
    return avgloss


# V. Training (Using Gradient Descent)
    
#i. Initializing epsilon for GD
eps = 0.05

print("\n")
print("\n")
print("\n")


print(f"Starting Training 2 \n")

print(f"Initial Loss: {loss(theta, X2, Y2):.4f} \n")

#ii. Initializing LR and Epochs
epochs = 150
lr = 1.0000

#iii. Gradient Descent Algorithm
for epoch in range(epochs):
    grad = np.zeros_like(theta) #Initializing Gradient as 0
        
    # Calculating numerical gradient for each parameter
    for i in range(len(theta)):
        thetaplus = theta.copy()
        thetaplus[i] += eps
        lossplus = loss(thetaplus, X2, Y2)
            
        thetaminus = theta.copy()
        thetaminus[i] -= eps
        lossminus = loss(thetaminus, X2, Y2)
            
        grad[i] = (lossplus - lossminus) / (2 * eps)
            
    #Update parameters (Gradient Descent step)
    theta -= lr * grad
        
    # Log progress
    if (epoch + 1) % 20 == 0 or epoch == 0:
        currentloss = loss(theta, X2, Y2)
        print(f"Epoch {epoch+1:3d}/{epochs} & Loss: {currentloss:.4f}")

# VI. Results
print("\n")
print("Final Results (for XOR Classification)")
print("\n")
    
finalhits = 0
for i in range(len(X2)):
    p = prob(X2[i], theta)
    pred = 1 if p > 0.5 else 0
    if pred == Y2[i]: finalhits += 1

    print(f"Input: {X2[i]} , Prob(1): {p:.4f}")
    print(f"Pred: {pred} , Actual: {Y2[i]}")
    print("\n")

print(f"Model Accuracy: {finalhits / len(X2) * 100:.4f}%")


