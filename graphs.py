import pandas as pd
import seaborn as sns
df = pd.DataFrame()
df["Episode Number"] = np.arange(1,len(plotinst.plotting_rewards)+1)
df["Rewards"] = plotinst.plotting_rewards
df["Average Rewards"] = np.cumsum(plotinst.plotting_rewards) / np.arange(1,len(plotinst.plotting_rewards)+1)
df["Actions"] = plotinst.total_actions
#pandas.DataFrame.to_csv()

#rewards graph
sns.lineplot(data = df, x = "Episode Number", y = "Rewards")
#average rewards graph
sns.lineplot(data = df, x = "Episode Number", y = "Average Rewards")
#both rewards on the same plot
sns.lineplot(data = df, x = "Episode Number", y = "Rewards")
sns.lineplot(data = df, x = "Episode Number", y = "Average Rewards")
#histogram of actions in the first episode
sns.histplot(df["Actions"][0], discrete=True)
#histogram of actions in the last episode
sns.histplot(df["Actions"][len(df["Actions"])-2], discrete=True)