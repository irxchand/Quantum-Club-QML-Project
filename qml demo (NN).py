import torch
import torch.nn as nn
import torch.optim as optim
import time

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

def run_experiment(optimizer_name):
    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.Tanh(),
        nn.Linear(2, 1),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()

    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.1)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.1)

    threshold = 1e-6
    max_epochs = 20000

    start = time.time()

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if loss.item() < threshold:
            break

    end = time.time()

    return epoch+1, loss.item(), end-start


# Run both
sgd = run_experiment("SGD")
adam = run_experiment("Adam")

print("FINAL COMPARISON")
print(f"SGD  -> Epochs: {sgd[0]}, Loss: {sgd[1]:.8f}, Time: {sgd[2]:.5f}s")
print(f"Adam -> Epochs: {adam[0]}, Loss: {adam[1]:.8f}, Time: {adam[2]:.5f}s")