import sklearn
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

n_samples = 1000

X,y = make_circles(n_samples, noise=0.03, random_state=42)

print(f"First 5 samples of X: \n {X[:5]} ")
print(f"First 5 samples of y: \n {y[:5]} ")

#I think this is showing that the inner most circle has a label of blue (0)
# And the outer red circle has a label of red
plt.scatter(x=X[:,0],
            y=X[:,1],
            c=y,
            cmap=plt.cm.RdYlBu)

#Lets check the shape of our features and labels
X.shape, y.shape

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

#View the first five samples
X[:5], y[:5]

X_train, X_test, y_train,y_test = train_test_split( X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

#In order to activate this we need to enable GPU access VIA plugin
device = "cude" if torch.cuda.is_available() else "cpu"

#We are going to implement supervised learning because we have features and labels and we are trying to link them up

class CircleModelV0(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.layer_1(x)) # computation goes through layer 1 and then layer 2
    
model_0 = CircleModelV0().to(device)

#We can also do the same thing by using nn.Sequential
model_1 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

model_1

#Sequential is good and all but it only runs in the sequence provided
    #This is why it is normally limited

untrained_preds = model_0(X_test)

print(f"The first 10 predictions are: {untrained_preds[:10,0]} \n")
print(f"The first 10 test results were: {y_test[:10]} \n")

#Need to use a different loss function because the one we originally chose ONLY works on regression problems

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)
