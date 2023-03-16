import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

#IN FUTURE Web Scrape the NBA data and find out which player has been increasingly doing good
    #Could help with placing bets on whether he will shoot 50% above and etc.

df = pd.read_excel("sportsref_klaystats.xlsx", usecols='L:AD')

nbaStatline = torch.from_numpy(df.to_numpy()[:,:18]).type(torch.float)
nbaLabel = torch.from_numpy(df.to_numpy()[:,18:]).type(torch.float)

nbaStatLineTrain = nbaStatline[:110,:]
nbaStatLineTest = nbaStatline[110:,:]

nbaLabelTrain = nbaLabel[:110].squeeze()
nbaLabelTest = nbaLabel[110:].squeeze()

class nbaPlayerDefineModel(nn.Module):
    """Attempting to predict whether NBA Player did Good/Bad in a certain night"""
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=18, out_features=180)
        self.layer_2 = nn.Linear(in_features=180, out_features=100)
        self.layer_3 = nn.Linear(in_features=100, out_features=40)
        self.layer_4 = nn.Linear(in_features=40, out_features=20)
        self.layer_5 = nn.Linear(in_features=20, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.layer_5(self.layer_4(self.relu(self.layer_3(self.relu(self.layer_2(self.layer_1(x))))))))

def accuracy_fn(y_true, y_pred):
    ratio = torch.eq(y_true,y_pred).sum().item()
    totalRatio = ratio/len(y_pred) * 100
    return totalRatio

model_0 = nbaPlayerDefineModel()

torch.manual_seed(42)
epochs = 801

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

epoch_count = []
accuracy_nums = []
loss_nums = []

for epoch in range(epochs):
    model_0.train()

    y_logits = model_0(nbaStatLineTrain).squeeze()
    y_pred = torch.round(y_logits)

    loss = loss_fn(y_logits, nbaLabelTrain)
    acc = accuracy_fn(y_true=nbaLabelTrain,y_pred=y_pred)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        # 1. Forward Pass
        test_logits = model_0(nbaStatLineTest).squeeze()
        test_pred = torch.round(test_logits)
        
        # 2. Calculate loss and accuracy
        test_loss = loss_fn(test_logits,nbaLabelTest)
        test_acc = accuracy_fn(y_true=nbaLabelTest,y_pred=test_pred)

    if epoch % 200 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
        epoch_count.append(epoch)
        accuracy_nums.append(acc)
        loss_nums.append(loss.cpu().detach().numpy())

model_0.eval()
with torch.inference_mode():
    y_preds = torch.round(model_0(nbaStatLineTest).squeeze())

print(y_preds)
print(nbaLabelTest)

model_0.eval()
def plotAllFeatures(feature1,
                    label1,
                    xLabel,
                    yLabel,
                    plotLabel):
    plt.plot(feature1, label1, c="b", label=plotLabel)
    plt.legend(prop={"size": 14})
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.show()

plotAllFeatures(epoch_count, accuracy_nums, "Epoch Count", "Accuracy Percentage", "accuracy")
plotAllFeatures(epoch_count, loss_nums, "Epoch Count", "Losses", "losses")

