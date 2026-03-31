# Intro to QML: Building a Classifier based on a Parameterized Quantum Circuit (PQC)
# This script demonstrates how to train a quantum circuit to classify non-linear data.

# Prerequisites
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ==========================================
# Demonstration 1: XOR Gate Output Prediction
# ==========================================

# Why XOR? The XOR problem is inherently non-linear. Classical linear models cannot solve it.
# By solving XOR, we demonstrate that our Quantum Machine Learning (QML) model can 
# successfully capture non-linear decision boundaries.

# I. Dataset Preparation
# We map the truth table of an XOR gate into numerical arrays for our model.

# i. Input features: The 4 possible states of a 2-bit system.
X1 = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

# ii. Target labels: XOR outputs (1 if inputs are different, 0 if they are the same).
Y1 = np.array([0, 1, 1, 0])

# iii. Initializing trainable parameters (Theta)
# We need 6 parameters for our specific PQC architecture (3 layers of rotations for 2 qubits).
# These are randomly initialized between 0 and 2*pi (a full rotation).
theta = np.random.uniform(0, 2 * np.pi, 6) 


# II. Quantum Circuit Definition (The "Model")
# This function constructs the PQC for a single data point (x1) and a given set of parameters (theta).
def circuit_initialization(x1, theta):

    # We use 2 qubits because our input data has 2 features.
    qc = QuantumCircuit(2) 
   
    # --- Data Encoding (Angle Encoding) ---
    # We embed classical data into the quantum state using Ry rotations.
    # Multiplying by pi scales the [0, 1] boolean inputs to [0, pi] angles.
    qc.ry(x1[0] * np.pi, 0) # Encode feature 1 into Qubit 0
    qc.ry(x1[1] * np.pi, 1) # Encode feature 2 into Qubit 1

    # --- Parameterized Quantum Circuit (Ansatz) ---
    # We build an entangling, parameterized architecture (Hardware Efficient Ansatz)
    
    # Layer 1: Trainable Rotations
    qc.ry(theta[0], 0) 
    qc.ry(theta[1], 1) 

    # Layer 2: Entanglement
    # The CNOT (cx) gate creates correlation between the qubits, crucial for capturing non-linear logic.
    qc.cx(0, 1) 

    # Layer 3: Trainable Rotations
    qc.ry(theta[2], 0) 
    qc.ry(theta[3], 1) 

    # Layer 4: Entanglement
    qc.cx(0, 1)

    # Layer 5: Final Trainable Rotations
    qc.ry(theta[4], 0) 
    qc.ry(theta[5], 1) 

    return qc


# III. Output Measurement and Probability Calculation
# This function runs the circuit and extracts the probability of predicting class '1'.
def prob(x1, theta):

    # i. Build the circuit for the current data point and parameters
    qc = circuit_initialization(x1, theta)

    # ii. Simulate the circuit to get the ideal Statevector (mathematical representation of the state)
    sv = Statevector.from_instruction(qc) 

    # iii. Extract the measurement probabilities for all 4 possible computational basis states:
    # |00> (index 0), |01> (index 1), |10> (index 2), |11> (index 3)
    probs = sv.probabilities()

    # iv. Define the probability of predicting "Class 1"
    # We map "Class 1" to the states where exactly one qubit is 1 (|01> and |10>).
    prob = probs[1] + probs[2] 

    return prob


# IV. Cost/Loss Function (Mean Squared Error)
# Calculates how far off the model's predictions are from the actual target labels.
def loss(theta, X1, Y1):
    
    total = 0 

    # i. Loop through every sample in the dataset
    for i in range(len(X1)):

        # Get the model's predicted probability for the current sample
        p = prob(X1[i], theta) 

        # Calculate squared error for this sample: (Predicted - Actual)^2
        total += (p - Y1[i])**2  

    # ii. Average the total error over the number of samples
    avgloss = (total) / (len(X1)) 
    return avgloss


# V. Model Training (Gradient Descent via Finite Differences)
    
# i. Hyperparameter: Epsilon is the small step size used to numerically approximate the gradient.
eps = 0.05

print("\n")

print(f"Starting Training 1\n")

print(f"Initial Loss: {loss(theta, X1, Y1):.4f} \n")

# ii. Hyperparameters: Learning Rate (step size for parameter updates) and Epochs (training loops)
epochs = 150
lr = 1.0000

# iii. The Training Loop
for epoch in range(epochs):
    
    # Array to hold the gradient (derivatives) for each of the 6 parameters
    grad = np.zeros_like(theta) 
        
    # Numerically calculate the gradient using the central difference method
    for i in range(len(theta)):
        # 1. Shift parameter up by epsilon and calculate loss
        thetaplus = theta.copy()
        thetaplus[i] += eps
        lossplus = loss(thetaplus, X1, Y1)
            
        # 2. Shift parameter down by epsilon and calculate loss
        thetaminus = theta.copy()
        thetaminus[i] -= eps
        lossminus = loss(thetaminus, X1, Y1)
            
        # 3. Approximate the derivative (slope)
        grad[i] = (lossplus - lossminus) / (2 * eps)
            
    # iv. Update the parameters by moving in the opposite direction of the gradient
    theta -= lr * grad
        
    # v. Log the progress every 20 epochs
    if (epoch + 1) % 20 == 0 or epoch == 0:
        currentloss = loss(theta, X1, Y1)
        print(f"Epoch {epoch+1:3d}/{epochs} & Loss: {currentloss:.4f}")

# VI. Evaluation and Results
# Testing the trained model on the same dataset to see how well it learned.
print("\n")
print("Final Results (for XOR Classification)")
print("\n")
    
finalhits = 0
for i in range(len(X1)):
    # Get predicted probability
    p = prob(X1[i], theta)
    
    # Threshold at 0.5: if probability > 50%, predict 1, else predict 0
    pred = 1 if p > 0.5 else 0
    
    # Check if prediction matches the actual label
    if pred == Y1[i]: finalhits += 1

    print(f"Input: {X1[i]} , Prob(1): {p:.4f}")
    print(f"Pred: {pred} , Actual: {Y1[i]}")
    print("\n")

print(f"Model Accuracy: {finalhits / len(X1) * 100:.4f}%")



# ==========================================
# Demonstration 2: 2D Point Output Prediction
# ==========================================

# This demonstration repeats the exact same methodology but frames the problem as predicting
# the class of 2D coordinates. The underlying logic and data shape remain non-linear.

# I. Dataset Preparation for 2D Point Prediction
X2 = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
Y2 = np.array([0, 1, 1, 0])

# Re-initializing trainable parameters (Theta) for the new model
theta = np.random.uniform(0, 2 * np.pi, 6) 


# II. Quantum Circuit Definition
# Identical architecture to Demo 1, but processes X2 data.
def circuit_initialization(x2, theta):

    qc = QuantumCircuit(2) 
   
    # Data Encoding (Angle Encoding into Ry gates)
    qc.ry(x2[0] * np.pi, 0) 
    qc.ry(x2[1] * np.pi, 1) 

    # Layer 1
    qc.ry(theta[0], 0) 
    qc.ry(theta[1], 1) 

    # Entanglement
    qc.cx(0, 1) 

    # Layer 2
    qc.ry(theta[2], 0) 
    qc.ry(theta[3], 1) 

    # Entanglement
    qc.cx(0, 1)

    # Layer 3
    qc.ry(theta[4], 0) 
    qc.ry(theta[5], 1) 

    return qc


# III. Output Measurement and Probability Calculation
def prob(x2, theta):

    qc = circuit_initialization(x2, theta)
    sv = Statevector.from_instruction(qc) 
    probs = sv.probabilities()

    # Summing probabilities of states |01> and |10> to represent Class 1
    prob = probs[1] + probs[2] 

    return prob


# IV. Cost/Loss Function (Mean Squared Error)
def loss(theta, X2, Y2):
    
    total = 0 
    for i in range(len(X2)):
        p = prob(X2[i], theta) 
        total += (p - Y2[i])**2  
    avgloss = (total) / (len(X2)) 
    return avgloss


# V. Model Training (Gradient Descent via Finite Differences)
eps = 0.05

print("\n")
print("\n")
print("\n")

print(f"Starting Training 2 \n")

print(f"Initial Loss: {loss(theta, X2, Y2):.4f} \n")

epochs = 150
lr = 1.0000

for epoch in range(epochs):
    grad = np.zeros_like(theta) 
        
    # Central difference numerical gradient
    for i in range(len(theta)):
        thetaplus = theta.copy()
        thetaplus[i] += eps
        lossplus = loss(thetaplus, X2, Y2)
            
        thetaminus = theta.copy()
        thetaminus[i] -= eps
        lossminus = loss(thetaminus, X2, Y2)
            
        grad[i] = (lossplus - lossminus) / (2 * eps)
            
    # Update parameters
    theta -= lr * grad
        
    if (epoch + 1) % 20 == 0 or epoch == 0:
        currentloss = loss(theta, X2, Y2)
        print(f"Epoch {epoch+1:3d}/{epochs} & Loss: {currentloss:.4f}")

# VI. Evaluation and Results
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