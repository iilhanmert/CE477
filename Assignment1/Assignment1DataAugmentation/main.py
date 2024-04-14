import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



np.random.seed(42)
original_data = {
    'age': np.random.randint(20, 80, 100),
    'bmi': np.random.uniform(18, 35, 100)
}
df = pd.DataFrame(original_data)

plt.figure(figsize=(10, 5))
plt.scatter(df['age'], df['bmi'], alpha=0.5)
plt.title('Original Data')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.grid(True)
plt.show()

noise = np.random.normal(0, 0.1, size=df.shape)
df_noisy = df + noise

plt.figure(figsize=(10, 5))
plt.scatter(df_noisy['age'], df_noisy['bmi'], alpha=0.5)
plt.title('Noisy Data (Data Augmentation)')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.grid(True)
plt.show()

angles = np.random.uniform(-10, 10, size=df.shape[0])
df_rotated = df.apply(lambda x: np.roll(x, np.random.randint(len(x))), axis=0)

plt.figure(figsize=(10, 5))
plt.scatter(df_rotated['age'], df_rotated['bmi'], alpha=0.5)
plt.title('Rotated Data (Data Augmentation)')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.grid(True)
plt.show()
