import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from praktika1 import X_train, X_test, y_train, y_test
import time

# Конвертируем данные в тензоры PyTorch
X_train_tensor = torch.FloatTensor (X_train)
y_train_tensor = torch.FloatTensor (y_train).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

# Создаём даталоадер
train_dataset = TensorDataset (X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Определяем модель
class SimpleNN (nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, 1)
        self.relu= nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Цикл обучения и замер времени
start_time = time.perf_counter()
epochs = 100
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model (batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
sum(range(10000000))
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Время обучения: {execution_time:.6f} секунд")

# Тестирование модели
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predicted = (test_outputs > 0.5).float()
    accuracy = (predicted == y_test).float().mean()
    print(f"Тoчность PyTorch: {accuracy:.4f}")