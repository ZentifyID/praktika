import numpy as np
import csv

np.random.seed(7)

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

input_size = 2
hidden_size = 4
output_size = 1

learning_rate = 0.5
momentum = 0.9
max_epochs = 10000
target_mse = 0.01

W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
b1 = np.zeros((1, hidden_size))
W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
b2 = np.zeros((1, output_size))

vW1 = np.zeros_like(W1)
vb1 = np.zeros_like(b1)
vW2 = np.zeros_like(W2)
vb2 = np.zeros_like(b2)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(a):
    return a * (1.0 - a)

history = []
stop_epoch = max_epochs

for epoch in range(1, max_epochs + 1):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    error = y - a2
    mse = np.mean(error ** 2)
    history.append((epoch, float(mse)))

    if mse < target_mse:
        stop_epoch = epoch
        break

    delta2 = error * dsigmoid(a2)
    delta1 = (delta2 @ W2.T) * dsigmoid(a1)

    gW2 = a1.T @ delta2 / len(X)
    gb2 = np.mean(delta2, axis=0, keepdims=True)
    gW1 = X.T @ delta1 / len(X)
    gb1 = np.mean(delta1, axis=0, keepdims=True)

    vW2 = momentum * vW2 + learning_rate * gW2
    vb2 = momentum * vb2 + learning_rate * gb2
    vW1 = momentum * vW1 + learning_rate * gW1
    vb1 = momentum * vb1 + learning_rate * gb1

    W2 += vW2
    b2 += vb2
    W1 += vW1
    b1 += vb1

a1 = sigmoid(X @ W1 + b1)
pred = sigmoid(a1 @ W2 + b2)

with open("xor_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["Вход 1", "Вход 2", "Ожидаемый выход", "Фактический выход", "Ошибка"])
    for i in range(len(X)):
        writer.writerow([int(X[i, 0]), int(X[i, 1]), int(y[i, 0]), f"{pred[i, 0]:.6f}", f"{abs(y[i, 0] - pred[i, 0]):.6f}"])

np.savez("xor_model.npz", W1=W1, b1=b1, W2=W2, b2=b2)

print(f"Обучение завершено на эпохе: {stop_epoch}")
print(f"Финальная MSE: {np.mean((y - pred) ** 2):.6f}")
print("\nРезультаты:")
for i in range(len(X)):
    print(f"{tuple(map(int, X[i]))} -> ожидается {int(y[i,0])}, получено {pred[i,0]:.6f}")
