import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets.cifar10 import load_data

(x_train, y_train), (x_test, y_test) = load_data()

print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(x_test))
print(np.shape(y_test))

# airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

print(y_train[111])
plt.imshow(x_train[130])
plt.show()
