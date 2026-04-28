import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# create circuit for n features
def make_circ(x, theta):
    n = len(x)
    qc = QuantumCircuit(n)

    for i in range(n):
        qc.ry(x[i] * np.pi, i)

    for i in range(n):
        qc.ry(theta[i], i)

    for i in range(n - 1):
        qc.cx(i, i + 1)

    for i in range(n):
        qc.ry(theta[n + i], i)

    return qc

# calc probabilities
def get_prob(x, theta):
    qc = make_circ(x, theta)
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities()

    prb_1 = 0
    n = len(x)

    for i in range(len(probs)):
        bitstring = format(i, '0%db' % n)
        if bitstring.count('1') % 2 == 1:
            prb_1 += probs[i]

    return prb_1

# loss function
def get_loss(theta, X, Y):
    tot = 0
    for i in range(len(X)):
        p = get_prob(X[i], theta)
        tot += (p - Y[i])**2
    return tot / len(X)

# train model
def train(X, Y, epochs=500, lr=0.1, eps=0.05):
    n = len(X[0])
    theta = np.random.uniform(0, 2*np.pi, 2*n)

    print("\nStarting Training\n")
    print("Initial Loss: %.4f\n" % get_loss(theta, X, Y))

    for e in range(epochs):
        g = np.zeros_like(theta)

        # gradient step
        for i in range(len(theta)):
            t_p = theta.copy()
            t_p[i] += eps
            l_p = get_loss(t_p, X, Y)

            t_m = theta.copy()
            t_m[i] -= eps
            l_m = get_loss(t_m, X, Y)

            g[i] = (l_p - l_m) / (2 * eps)

        theta -= lr * g

        # log
        if (e + 1) % 20 == 0 or e == 0:
            c_loss = get_loss(theta, X, Y)
            print("Epoch %3d/%d | Loss: %.4f" % (e+1, epochs, c_loss))

    return theta


def predict(x, theta):
    p = get_prob(x, theta)
    return (1 if p > 0.5 else 0), p


if __name__ == "__main__":
    np.random.seed(42)

    features = int(input("Enter number of features: "))
    samples = int(input("Enter number of samples: "))

    X = []
    Y = []

    print("\n--- ENTER TRAINING DATA ---\n")

    for i in range(samples):
        while True:
            x_input = input("Sample %d (enter %d values comma-separated): " % (i+1, features))
            try:
                x = list(map(float, x_input.split(",")))
                if len(x) != features:
                    print("Invalid input size.")
                    continue
                break
            except:
                print("Invalid format.")

        while True:
            y_input = input("Label for sample %d (0 or 1): " % (i+1))
            if y_input in ["0", "1"]:
                y = int(y_input)
                break
            else:
                print("Invalid label.")

        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)

    select_options = input("Do you want to change the custom epochs, learning rate, or epsilon? (yes/no): ")
    if select_options.lower() == "yes":
        epochs = int(input("Enter number of epochs(std is 500): "))
        lr = float(input("Enter learning rate(std is 0.1): "))
        eps = float(input("Enter epsilon(std is 0.05): "))
    else:
        epochs = 500
        lr = 0.1
        eps = 0.05

    theta = train(X, Y, epochs, lr, eps)

    print("\nTraining Complete\n")

    print("PREDICTIONS\n")

    while True:
        user_input = input("Enter input (%d values) or 'exit': " % features)
        
        if user_input.lower() == "exit":
            break

        try:
            x = np.array(list(map(float, user_input.split(","))))

            if len(x) != features:
                print("Invalid input size.")
                continue

            pred, p = predict(x, theta)

            print("Prob(1): %.4f" % p)
            print("Prediction: %d\n" % pred)

        except Exception as e:
            print("Invalid input.")
