#Intro to QML: Building a Classifier based on PQC:

#Prerequisties
import numpy as np
import time
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

def make_qc(x1, theta):

    #x is the input data (X1[i])
    #theta is the training parameters (theta[i])

    qc = QuantumCircuit(2) #As there are 2 features (X1[0][0] and X1[0][1] (For first element, 0, 0 as first and second input respectively)), we need 2 qubits (as one qubit holds one features/gates)
   
    #Defining conversion of classical data to quantum states

    #Syntax for encoding data: qc.gate(operation, qubit)

    #Input Data to Rotation in Space
    qc.ry(x1[0] * np.pi, 0) #Encoding the state of probe 1 (as there are 2 probes on a gate) as [0](first input (in (x1, x2)))
    qc.ry(x1[1] * np.pi, 1) #Encoding the state of probe 2 as [1](second input (in (x1, x2)))

    #Step 1: Training Parameters (accessing the first and second element of theta and rotating accordingly)
    qc.ry(theta[0], 0) #First training parameter
    qc.ry(theta[1], 1) #Second training parameter

    #Step 2: Entangling 2 qubits
    qc.cx(0, 1) 

    #Step 3: Training Parameters (accessing the third and fourth element of theta and rotating accordingly)
    qc.ry(theta[2], 0) #Third training parameter
    qc.ry(theta[3], 1) #Fourth training parameter

    #Step 4: Re Entangling 2 qubits
    qc.cx(0, 1)

    #Step 5: Training Parameters (accessing the fifth and sixth element of theta and rotating accordingly)
    qc.ry(theta[4], 0) #Fifth training parameter
    qc.ry(theta[5], 1) #Sixth training parameter

    #Through this, 3 iterations of Rotation would be done, before displaying the output.

    return qc


# III. Finding Probability for Each Input (using as function as it will be iterated for each input)

def prob(x1, theta):

    #i. Getting Circuit for Inputs (based off the value of i in for loop)
    qc = make_qc(x1, theta)

    #ii. Getting Outputs as Statevector
    sv = Statevector.from_instruction(qc) #from_instruction is qiskit function

    #iii. Getting Probabilities from Statevector
    probs = sv.probabilities()

    #iv. Getting Probability of output (y) of 1
    prb = probs[1] + probs[2] #probs[0] is 0,0, probs[1] is 0,1, probs[2] is 1,0, probs[3] is 1,1

    #v. Returning Probability
    return prb


# IV. Computing Loss (MSE) (as a function as it would be iterated as a function for each input)

def get_loss(theta, X1, Y1):
    
    #i. Initializing total loss as 0
    tot = 0 

    #ii. Iterating for each input
    for i in range(len(X1)):

        #Getting probability for each input (calling prob fxn, passing X1[i] and random theta, getting prob for y = 1 for that input as p)
        p = prob(X1[i], theta) 

        #Calculating loss
        tot += (p - Y1[i])**2  # Predicted - Real (sq)

    #iii. Returning Avg Loss
    avg_l = (tot) / (len(X1)) 
    return avg_l


# V. Training (Using Gradient Descent)
    
#i. Initializing epsilon for GD
eps = 0.05

print("\n")

print("Starting Training 1\n")

print("Initial Loss: %.4f \n" % get_loss(theta, X1, Y1))

#ii. Initializing LR and Epochs
epochs = 60
lr = 1.5000

#iii. Gradient Descent Algorithm
start_time_1 = time.time()
for e in range(epochs):
    g = np.zeros_like(theta) #Initializing Gradient as 0
        
    # Calculating numerical gradient for each parameter
    for i in range(len(theta)):
        t_p = theta.copy()
        t_p[i] += eps
        l_p = get_loss(t_p, X1, Y1)
            
        t_m = theta.copy()
        t_m[i] -= eps
        l_m = get_loss(t_m, X1, Y1)
            
        g[i] = (l_p - l_m) / (2 * eps)
            
    #Update parameters (Gradient Descent step)
    theta -= lr * g
        
    # Log progress
    if (e + 1) % 20 == 0 or e == 0:
        c_loss = get_loss(theta, X1, Y1)
        print("Epoch %3d/%d & Loss: %.4f" % (e+1, epochs, c_loss))

end_time_1 = time.time()
print("\nTraining 1 completed in %.2f seconds." % (end_time_1 - start_time_1))

# VI. Results
print("\n")
print("Final Results (for XOR Classification)")
print("\n")
    
hits = 0
for i in range(len(X1)):
    p = prob(X1[i], theta)
    pred = 1 if p > 0.5 else 0
    if pred == Y1[i]: hits += 1

    print("Input: [%d, %d] , Prob(1): %.4f" % (X1[i][0], X1[i][1], p))
    print("Pred: %d , Actual: %d\n" % (pred, Y1[i]))

print("Model Accuracy: %.4f%%\n" % (hits / len(X1) * 100))



#2. 2D Point Output Prediction using PQC:

# Using 2D Point Prediction as the SECOND QML demo using PQC.
# As, 2D Point Prediction is Non Linear (as a set of non linear points are chosen), Thus the prediction pattern is also Non Linear. Hence, being used for simplicity of the second demo.
# For 2D Point Prediction, (0,3) , (2,2) are 1 and (1,1) , (4,14) are 0. Hence, Encoding those Rules as Data.

# I. Data for Model / Rules of 2D Point Prediction in Numerical Form

#i. Inputs for 2D Point Prediction (Normalized independently per feature)
X2 = np.array([ [0/4, 3/14], [1/4, 1/14], [4/4, 2/14], [2/4, 14/14] ])

#ii. Outputs for 2D Point Prediction (Binary Labels: 1 for odd parity, 0 for even)
Y2 = np.array([1, 0, 1, 0])

#iii. Input as x2
#Defined as x2 = X2[i] using for loop in the argument of function prob()

#iv. Initial Parameters (Theta)
theta = np.random.uniform(0, 2 * np.pi, 12) #Increased to 12 for RY+RX layers


# II. Defining Circuit as Function (as it can only run for 1 input of X1 (for 2 pts as 2 qubits) at a time, this should be called 4 times)

def make_qc_2d(x2, theta):

    #x is the input data (X1[i])
    #theta is the training parameters (theta[i])

    qc = QuantumCircuit(2) #As there are 2 features (X1[0][0] and X1[0][1] (For first element, 0, 0 as first and second input respectively)), we need 2 qubits (as one qubit holds one features/pts)
   
    #Defining conversion of classical data to quantum states

    #Data Re-uploading structure: interleaved data + parameter encoding
    
    #Block 1
    qc.ry(x2[0] * np.pi * 2 + theta[0], 0)
    qc.ry(x2[1] * np.pi * 2 + theta[1], 1)
    qc.rz(theta[2], 0)
    qc.rz(theta[3], 1)
    qc.cx(0, 1)

    #Block 2
    qc.ry(x2[0] * np.pi * 2 + theta[4], 0)
    qc.ry(x2[1] * np.pi * 2 + theta[5], 1)
    qc.rz(theta[6], 0)
    qc.rz(theta[7], 1)
    qc.cx(0, 1)

    #Block 3
    qc.ry(x2[0] * np.pi * 2 + theta[8], 0)
    qc.ry(x2[1] * np.pi * 2 + theta[9], 1)
    qc.rz(theta[10], 0)
    qc.rz(theta[11], 1)

    #Through this, multiple iterations of Rotation would be done, acting as a universal approximator.

    return qc


# III. Finding Probability for Each Input (using as function as it will be iterated for each input)

def prob_2d(x2, theta):

    #i. Getting Circuit for Inputs (based off the value of i in for loop)
    qc = make_qc_2d(x2, theta)

    #ii. Getting Outputs as Statevector
    sv = Statevector.from_instruction(qc) #from_instruction is qiskit function

    #iii. Getting Probabilities from Statevector
    probs = sv.probabilities()

    #iv. Getting Probability of output (y) of 1 (using basis parity or just prob(1))
    prb = probs[1] + probs[3] #probs[0]=00, probs[1]=01, probs[2]=10, probs[3]=11. Target last qubit=1.

    #v. Returning Probability
    return prb


# IV. Computing Loss (MSE) (as a function as it would be iterated as a function for each input)

def get_loss_2d(theta, X2, Y2):
    
    #i. Initializing total loss as 0
    tot = 0 

    #ii. Iterating for each input
    for i in range(len(X2)):

        #Getting probability for each input (calling prob fxn, passing X1[i] and random theta, getting prob for y = 1 for that input as p)
        p = prob_2d(X2[i], theta) 

        #Calculating loss
        tot += (p - Y2[i])**2  # Predicted - Real (sq)

    #iii. Returning Avg Loss
    avg_l = (tot) / (len(X2)) 
    return avg_l


# V. Training (Using Gradient Descent)
    
#i. Initializing epsilon for GD (Fine precision for deep data re-uploading)
eps = 0.01

print("\n")
print("\n")
print("\n")


print("Starting Training 2 \n")

print("Initial Loss: %.4f \n" % get_loss_2d(theta, X2, Y2))

#ii. Initializing LR and Epochs
lr = 2.5000
epochs = 120

#iii. Gradient Descent Algorithm
start_time_2 = time.time()
for e in range(epochs):
    g = np.zeros_like(theta) #Initializing Gradient as 0
        
    # Calculating numerical gradient for each parameter
    for i in range(len(theta)):
        t_p = theta.copy()
        t_p[i] += eps
        l_p = get_loss_2d(t_p, X2, Y2)
            
        t_m = theta.copy()
        t_m[i] -= eps
        l_m = get_loss_2d(t_m, X2, Y2)
            
        g[i] = (l_p - l_m) / (2 * eps)
            
    #Update parameters (Gradient Descent step)
    theta -= lr * g
        
    # Log progress
    if (e + 1) % 20 == 0 or e == 0:
        c_loss = get_loss_2d(theta, X2, Y2)
        print("Epoch %3d/%d & Loss: %.4f" % (e+1, epochs, c_loss))

end_time_2 = time.time()
print("\nTraining 2 completed in %.2f seconds." % (end_time_2 - start_time_2))

# VI. Results
print("\n")
print("Final Results (for 2D Point Classification)")
print("\n")
    
hits = 0
preds = []
for i in range(len(X2)):
    p = prob_2d(X2[i], theta)
    pred = 1 if p > 0.5 else 0
    preds.append(pred)
    if pred == Y2[i]: hits += 1

    print("Input: [%.2f, %.2f] , Prob(1): %.4f" % (X2[i][0], X2[i][1], p))
    print("Pred: %d , Actual: %d\n" % (pred, Y2[i]))

print("Model Accuracy: %.4f%%\n" % (hits / len(X2) * 100))

# VII. Evaluation on Test Cases
print("\n")
print("Generating Noisy Test Cases for Evaluation...")
print("\n")

# Generating 40 test cases by adding noise to our known points
import numpy as np
np.random.seed(42)  # For reproducibility
X_test = []
Y_test = []

num_test_cases_per_point = 10
for _ in range(num_test_cases_per_point):
    for j in range(len(X2)):
        # Add slight gaussian noise
        noisy_point = X2[j] + np.random.normal(0, 0.08, 2)
        X_test.append(noisy_point)
        Y_test.append(Y2[j])

X_test = np.array(X_test)
Y_test = np.array(Y_test)

test_hits = 0
test_preds = []
for i in range(len(X_test)):
    p = prob_2d(X_test[i], theta)
    pred = 1 if p > 0.5 else 0
    test_preds.append(pred)
    if pred == Y_test[i]: test_hits += 1

print("Evaluation Accuracy on %d Test Cases: %.4f%%\n" % (len(X_test), (test_hits / len(X_test) * 100)))

# View Confusion Matrix for these Test Cases
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(Y_test, test_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Even (0)", "Odd (1)"], 
            yticklabels=["Even (0)", "Odd (1)"])
plt.title("Evaluation Confusion Matrix (Noisy Test Cases)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
