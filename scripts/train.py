from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np

# Set random seed
seed = 42

################################
########## DATA PREP ###########
################################

# Load in the data
df = pd.read_csv("data/features_data.csv")

# Split into train and test sections
diagnosis = df.pop("diagnosis")
# Label encode the string value: diagnosis
encoder = preprocessing.LabelEncoder()    
y = np.float64(encoder.fit_transform(diagnosis))

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=seed)

#################################
########## MODELLING ############
#################################

# Fit a model on the train section
regr = RandomForestRegressor(max_depth=5, random_state=seed)
regr.fit(X_train, y_train)

# Report training set score
train_score = regr.score(X_train, y_train) * 100
# Report test set score
test_score = regr.score(X_test, y_test) * 100

# Write scores to a file
with open("data/metrics.txt", 'w') as outfile:
        outfile.write("Training Score: %2.1f%%\n" % train_score)
        outfile.write("Test Score: %2.1f%%\n" % test_score)


##########################################
##### PLOT FEATURE IMPORTANCE ############
##########################################
# Calculate feature importance in random forest
importances = regr.feature_importances_
labels = df.columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ["feature","importance"])
feature_df = feature_df.sort_values(by='importance', ascending=False,)

# image formatting
axis_fs = 14 #fontsize
title_fs = 16 #fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel('Importance',fontsize = axis_fs) 
ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
ax.set_title('Random Forest Feature Importance', fontsize = title_fs)

plt.tight_layout()
plt.savefig("data/feature_importance.png",dpi=120) 
plt.close()
