import pandas as pd
import torch
import matplotlib.pyplot as plt

#IN FUTURE Web Scrape the NBA data and find out which player has been increasingly doing good
    #Could help with placing bets on whether he will shoot 50% above and etc.

df = pd.read_excel("NBAStats.xlsx", )
print(df)

nbaStatline = torch.from_numpy(df.to_numpy()[:,:8])
nbaLabel = torch.from_numpy(df.to_numpy()[:,8:])


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

