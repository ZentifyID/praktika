from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from praktika1 import X_train, X_test, y_train, y_test
import time

# Создаём модель
mlp = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)

# Обучаем модель и замеряем время
start_time = time.perf_counter()
mlp.fit(X_train, y_train)
sum(range(10000000))
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Время обучения: {execution_time:.6f} секунд")

# Делаем предсказания
y_pred = mlp.predict(X_test)

# Оцениваем точность
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность scikit-learn MLP: {accuracy:.4f}")

