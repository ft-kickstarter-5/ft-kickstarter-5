from category_encoders import OrdinalEncoder
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as ply
def wrangle(df):
  df = df.drop(columns=['id','usd_pledged','spotlight','photo','state','name',
                        'blurb', 'slug','Unnamed: 0','creator','location','profile','urls','source_url',
                        'friends','is_starred','is_backing','permissions','name_len','name_len_clean'])
  return df


df = pd.read_csv('/content/gdrive/MyDrive/Untitled folder/A/kickstarter_data_full.csv')
wrangle(df)

col = df.columns
test_col = df[['goal','pledged','launch_to_deadline_days','launch_to_state_change_days','backers_count','category','SuccessfulBool']]
pd.DataFrame(test_col)

target = 'SuccessfulBool'
#y= df[target]
#X = df.drop(columns=target)
y = test_col[target]
X = test_col.drop(columns=target)

X_train, X_val, y_train, y_val = train_test_split(X, y , test_size=0.2, random_state =7)

# Decision Tree
model_dt = make_pipeline(
    OrdinalEncoder(),
    SimpleImputer(strategy='mean'),
    DecisionTreeClassifier(random_state=7)
)

model_dt.fit(X_train, y_train)

#Random Forest
model_rf = make_pipeline(
    OrdinalEncoder(),
    SimpleImputer(),
    RandomForestClassifier(random_state=7, n_jobs = -1)
)

model_rf.fit(X_train, y_train)

y_train.value_counts(normalize=True).max()

print('Training Accuracy:', model_dt.score(X_train, y_train))
print('Validation Accuracy:', model_dt.score(X_val, y_val))

print('Training Accuracy:', model_rf.score(X_train, y_train))
print(' Validation Accuracy:', model_rf.score(X_val, y_val))

model_rf.score(X_val,y_val)

result = permutation_importance(model_rf, X_val, y_val, random_state=7)
forest_importances = pd.Series(result.importances_mean)
graph1 = forest_importances.plot.bar(y=forest_importances,x=col)
graph1.set_title("Feature importances using permutation on full model")
graph1.set_ylabel("Importance of features")
plt.show()