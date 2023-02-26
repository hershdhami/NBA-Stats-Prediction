import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

n_samples = 1000

X,y = make_circles(n_samples, noise=0.03, random_state=42)

print(f"First 5 samples of X: \n {X[:5]} ")
print(f"First 5 samples of y: \n {y[:5]} ")

plt.scatter(x=X[:,0],
            y=X[:,1],
            c=y,
            cmap=plt.cm.RdYlBu)