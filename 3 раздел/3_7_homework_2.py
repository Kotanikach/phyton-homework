import numpy as np
from tensorflow import keras

num_samples = 1000
x1 = np.random.randint(0, 11, num_samples)
x2 = np.random.randint(0, 11, num_samples)
X = np.column_stack([x1, x2])
y = x1 + x2

# Нормируем
X = X / 10.0

model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X, y, epochs=50, verbose=0)
loss, mae = model.evaluate(X, y, verbose=0)
print(f"Mean Absolute Error: {mae}")

x1_test = np.array([3, 5, 9])
x2_test = np.array([2, 5, 1])
X_test = np.column_stack([x1_test, x2_test]) / 10.0
predictions = model.predict(X_test)

print("Predictions:")
for i in range(len(x1_test)):
    print(f"{x1_test[i]} + {x2_test[i]} = {predictions[i][0]:.2f}")