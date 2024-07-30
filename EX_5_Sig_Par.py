import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

st.title('Sigmoid Function Visualizer with Binary Sample Points')

st.sidebar.header('Parameters')
w = st.sidebar.slider('w', -10.0, 10.0, 1.0)
b = st.sidebar.slider('b', -10.0, 10.0, 0.0)

x = np.linspace(-10, 10, 400)
z = w * x + b
y = sigmoid(z)

# Generate binary sample points (0 or 1)
np.random.seed(0)
sample_x = np.random.uniform(0, 10, 20)
#sample_y = (np.random.rand(20) > 0.5).astype(int)
sample_y = np.ones(20)

sample_x1 = np.random.uniform(-15, -5, 20)
sample_y1 = np.zeros(20)
#sample_y1 = (np.random.rand(20) > 0.5).astype(int)

fig, ax = plt.subplots()
ax.plot(x, y, label='Sigmoid function')
ax.scatter(sample_x, sample_y, color='red', label='Binary sample points')
ax.scatter(sample_x1, sample_y1, color='green', label='Binary sample points')

ax.set_xlabel('x')
ax.set_ylabel('σ(z)')
ax.set_title(f'Sigmoid Function with w={w} and b={b}')
ax.grid(True)
ax.legend()

st.pyplot(fig)