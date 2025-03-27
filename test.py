import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("data/clean_data.csv")
pd.set_option('display.max_columns', None)


# Inspect the DataFrame to understand its structure
print(df.head())

# Remove commas and convert 'No of Persons Voted' to numeric
# df["No of Persons Voted"] = df["No of Persons Voted"].str.replace(',', '').astype(float)

# # Create an lmplot to examine the relationship between 'No of Persons Voted' and 'Rating'
# sns.lmplot(data=df, x="No of Persons Voted", y="Rating", ci=None, height=6, aspect=1.5)
# plt.title("Linear Relationship Between No of Persons Voted and Rating")
# plt.xlabel("No of Persons Voted")
# plt.ylabel("Rating")
# plt.show()
