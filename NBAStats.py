import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

#IN FUTURE Web Scrape the NBA data and find out which player has been increasingly doing good
    #Could help with placing bets on whether he will shoot 50% above and etc.

df = pd.read_excel("NBAStats.xlsx", )
print(df)

nbaStatline = torch.from_numpy(df.to_numpy()[:,:8]).type(torch.float)
nbaLabel = torch.from_numpy(df.to_numpy()[:,8:]).type(torch.float)

nbaStatLineTrain = nbaStatline[:5,:]
nbaStatLineTest = nbaStatline[5:,:]

nbaLabelTrain = nbaLabel[:5].squeeze()
nbaLabelTest = nbaLabel[5:].squeeze()


#This displays the graphs compares to Good/Bad

# SCATTER PLOT OF EVERYTHING
# plt.scatter(x=nbaStatline[:,2],
#             y=nbaStatline[:,3],
#             c=nbaLabel,
#             cmap=plt.cm.RdYlBu)

def plotAllFeatures(feature1,
                    feature2,
                    feature3,
                    label):
    plt.scatter(nbaStatline[:,0], nbaLabel, c="b", s=4, label="Minutes")
    plt.scatter(nbaStatline[:,1], nbaLabel, c="g", s=4, label="Points")
    plt.scatter(nbaStatline[:,2], nbaLabel, c="r", s=4, label="FG")
    plt.legend(prop={"size": 14})

plotAllFeatures(nbaStatline[:,0], nbaStatline[:,1], nbaStatline[:,2], nbaLabel)

class nbaPlayerDefineModel(nn.Module):
    """Attempting to predict whether NBA Player did Good/Bad in a certain night"""
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=8, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.layer_2(self.relu(self.layer_1(x))))

def accuracy_fn(y_true, y_pred):
    ratio = torch.eq(y_true,y_pred).sum().item()
    totalRatio = ratio/len(y_pred) * 100
    return totalRatio

model_0 = nbaPlayerDefineModel()

torch.manual_seed(42)
epochs = 1000

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

print()

for epoch in range(epochs):
    model_0.train()

    y_logits = model_0(nbaStatLineTrain).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, nbaLabelTrain)
    acc = accuracy_fn(y_true=nbaLabelTrain,y_pred=y_pred)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        # 1. Forward Pass
        test_logits = model_0(nbaStatLineTest).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        # 2. Calculate loss and accuracy
        test_loss = loss_fn(test_logits,nbaLabelTest)
        test_acc = accuracy_fn(y_true=nbaLabelTest,y_pred=test_pred)

    if epoch % 200 == 0:
        print(y_pred)
        print(nbaLabelTrain)
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")


model_0.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_0(nbaStatLineTest).squeeze()))

y_preds, nbaLabelTest

#It is working but it is underfitted because of the lack of data!!!
