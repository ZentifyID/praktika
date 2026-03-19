import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from praktika1 import X_train, X_test, y_train, y_test
import time

# Создаём последовательную модель
model = Sequential([
    Dense (50, activation='relu', input_shape=(10,)),
    Dense (30, activation='relu'),
    Dense (1, activation='sigmoid')
])

# Компилируем модель
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучаем модель и замеряем время
start_time = time.perf_counter()
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)
sum(range(10000000))
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Время обучения: {execution_time:.6f} секунд")

# Оцениваем модель
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print (f"Тoчность TensorFlow/Keras: {test_accuracy: .4f}")


# Визуализация процесса обучения
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.show()