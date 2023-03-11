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

X_train, X_test, y_train,y_test = train_test_split( X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

#In order to activate this we need to enable GPU access VIA plugin
#device = "cude" if torch.cuda.is_available() else "cpu"

#We are going to implement supervised learning because we have features and labels and we are trying to link them up

class CircleModelV0(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.layer_1(x)) # computation goes through layer 1 and then layer 2
    
model_0 = CircleModelV0()

#We can also do the same thing by using nn.Sequential
model_1 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
)

model_1

#Sequential is good and all but it only runs in the sequence provided
    #This is why it is normally limited

untrained_preds = model_0(X_test)

print(f"The first 10 predictions are: {untrained_preds[:10,0]} \n")
print(f"The first 10 test results were: {y_test[:10]} \n")

#Need to use a different loss function because the one we originally chose ONLY works on regression problems

def accuracy_fn(y_true, y_pred):
    ratio = torch.eq(y_true,y_pred).sum().item()
    totalRatio = ratio/len(y_pred) * 100
    return totalRatio

#This model will not work because we are using only lines to predict a circular graph
    #We are underfitting the model meaning we need to learn predictive patterns from data
    #We are going to need to create a new entire model!


class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # <- add in ReLU activation function
        # Can also put sigmoid in the model 
        # This would mean you don't need to use it on the predictions
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      # Intersperse the ReLU activation function between layers
       return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
    
model_2 = CircleModelV2()

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.1)

torch.manual_seed(42)
epochs = 1000

for epoch in range(epochs):
    #A logit is a function that represents probability from 0 to 1
    y_logits = model_2(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train,y_pred=y_pred)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_2.eval()
    with torch.inference_mode():
        # 1. Forward Pass
        test_logits = model_2(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Calculate loss and accuracy
        test_loss = loss_fn(test_logits,y_test)
        test_acc = accuracy_fn(y_true=y_test,y_pred=test_pred)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

model_2.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_2(X_test).squeeze()))

y_preds[:10], y_test[:10]
