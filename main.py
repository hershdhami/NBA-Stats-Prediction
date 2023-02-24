import torch
from torch import nn # nn contains all of PyTorch's building blocks
import matplotlib.pyplot as plt

#Check PyTorch Version
torch.__version__

#Create *known* parameters
weight = 0.7
bias = 0.3

#Want to build a model that estimates these numbers
start = 0
end = 1
step = 0.02

#Typically x is a matrix or tensor
#Capital represent matrix or tensor
X = torch.arange(start, end, step).unsqueeze(dim = 1)
y = weight * X + bias

train_split = int(0.8 * len(X)) #80% of training set and 20% for testing
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):

  plt.figure(figsize=(10,7))

  #PLot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

  #Plot the test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")

  if predictions is not None:
    #Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  #Show the legend
  plt.legend(prop={"size": 14})

plot_predictions()

class LinearRegressionModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.weights = nn.Parameter(torch.randn(1, 
                                          dtype=torch.float),
                              requires_grad=True)
    
    self.bias = nn.Parameter(torch.randn(1,
                                        dtype=torch.float),
                            requires_grad=True)
    
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    return self.weights * x + self.bias


#Set the manual seed since nn.Paramater are randomly initialized
torch.manual_seed(42)

model_0 = LinearRegressionModel()

#Make predictions with model
with torch.inference_mode():
  y_preds = model_0(X_test)

print(f"Number of testing samples: {len(X_test)}")
print(f"The testing values are: \n {X_test}")
print(f"Predicted values: \n {y_preds}")

plot_predictions(predictions=y_preds)

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=1e-2)

#What is Stoichastic Gradient Descent?
#Loss Function (Could use the sum of the squared residuals from a random line of best fit 
# to determine how well that line fit into the data)
    # Note: Ln = |Xn - Yn|

#Residual is the different between observed and predicted values
  #The loss function will square these numbers and sum them

#Sets a seed for returning random numbers
torch.manual_seed (42)

epochs = 100

train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
  ### TRAINING MODE

  model_0.train()

  # 1. Forward pass on train data
  y_pred = model_0(X_train)

  # 2. Check loss of train data
  loss = loss_fn(y_pred, y_train)

  # 3. Zero grad of the optimizer
  optimizer.zero_grad()

  # 4. Loss Backwards
  loss.backward()

  # 5. Progress the optimizer
  optimizer.step()

  #TESTING MODE

  # Put the model in evaluation mode
  model_0.eval()

  with torch.inference_mode():
    # 1. Forward pass on test data
    test_pred = model_0(X_test)

    #Note when converting a tensor to an array make sure you have the torch.float property enabled
      #Otherwise an error will be signalled / occur
    test_loss = loss_fn(test_pred, y_test.type(torch.float))

    if epoch % 10 == 0:
      epoch_count.append(epoch)
      train_loss_values.append(loss.detach().numpy())
      test_loss_values.append(test_loss.detach().numpy())

print(f"The changing test_loss_values are: {test_loss_values}")

