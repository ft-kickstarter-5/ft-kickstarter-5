import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 67)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale = 1)
from matplotlib import inline
plt.style.use('ggplot')

np.random.seed(7)
from prettytable import PrettyTable
import warnings
warnings.filterwarnings('ignore')

from IPython.core.pylabtools import figsize

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE

from keras.models import Sequential
from keras import layers
from keras.layers import LSTM,Dropout
from keras.layers import Dense,Conv1D,MaxPooling1D
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences




kickstarter = pd.read_csv('../../kickstarter_data_full.csv', index_col=0)

cols_to_drop = ['friends', 'is_starred', 'is_backing', 'permissions'
                'id', 
                'photo', 
                'slug', 
                'currency_symbol',
                'currency_trailing_code', 
                'creator', 
                'location',
                'urls', 
                'source_url', 
                'name_len', 
                'blurb_len',
                'create_to_launch', 
                'launch_to_deadline', 
                'launch_to_state_change',
                'USorGB', 
                'TOPCOUNTRY', 
                'LaunchedTuesday', 
                'DeadlineWeekend',
                'deadline_month', 'deadline_day', 'deadline_yr', 'deadline_hr', 
                'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr',
                'created_at_month', 'created_at_day', 'created_at_yr', 'created_at_hr',
                'launched_at_month', 'launched_at_day', 'launched_at_yr', 'launched_at_hr'
                ]
kickstarter.drop(labels=cols_to_drop, axis=1, inplace=True)
kickstarter.drop(labels='profile', axis=1, inplace=True)


#converts type bool to 0 for false and 1 for true
kickstarter['disable_communication'] = kickstarter['disable_communication'] * 1 
#converts type bool to 0 for false and 1 for true
kickstarter['staff_pick'] = kickstarter['staff_pick'] * 1 
#converts type bool to 0 for false and 1 for true
kickstarter['spotlight'] = kickstarter['spotlight'] * 1 


first_quartile = kickstarter['goal'].describe()['25%']
third_quartile = kickstarter['goal'].describe()['75%']
iqr = third_quartile - first_quartile
kickstarter_goal_iqr = kickstarter[(kickstarter['goal'] > first_quartile) & (kickstarter['goal'] < third_quartile)]

kickstarter_iqr_trimmed = kickstarter_goal_iqr

first_quartile = kickstarter['create_to_launch_days'].describe()['25%']
third_quartile = kickstarter['create_to_launch_days'].describe()['75%']

iqr = third_quartile - first_quartile

kickstarter_iqr_trimmed = kickstarter[(kickstarter['create_to_launch_days'] > first_quartile) & (kickstarter['create_to_launch_days'] < third_quartile)]

first_quartile = kickstarter['pledged'].describe()['25%']
third_quartile = kickstarter['pledged'].describe()['75%']

iqr = third_quartile - first_quartile

kickstarter_iqr_trimmed = kickstarter[(kickstarter['pledged'] > first_quartile) & (kickstarter['pledged'] < third_quartile)]

first_quartile = kickstarter['backers_count'].describe()['25%']
third_quartile = kickstarter['backers_count'].describe()['75%']

iqr = third_quartile - first_quartile

kickstarter_iqr_trimmed = kickstarter[(kickstarter['backers_count'] > first_quartile) & (kickstarter['backers_count'] < third_quartile)]

reduced_x_features = kickstarter_iqr_trimmed[['launch_to_deadline_days', 'staff_pick', 'pledged', 'backers_count', 'spotlight', 'goal']]
reduced_y = kickstarter_iqr_trimmed[['SuccessfulBool']]

numeric_subset = kickstarter_iqr_trimmed.select_dtypes('number')

for col in numeric_subset.columns:
    if col == 'SuccessfulBool':
        next
    else:
        numeric_subset['sqrt_' + col] = np.sqrt(numeric_subset[col])
        numeric_subset['log_' + col] = np.log(numeric_subset[col])

categorical_subset = kickstarter_iqr_trimmed['category']

categorical_subset = pd.get_dummies(categorical_subset)
features = pd.concat([numeric_subset, categorical_subset], axis = 1)
features = features.dropna(subset = ['SuccessfulBool'])

correlations = features.corr()['SuccessfulBool'].dropna().sort_values()
#correlations.head()

reduced_x_features['log_goal'] = features['log_goal']
reduced_x_features['log_pledged'] = features['log_pledged']



kickstarter_X = []
kickstarter_y = []
for i, j in reduced_x_features.iterrows():
    tmp = str(reduced_x_features['launch_to_deadline_days'][i]) + " " + \
        str(reduced_x_features['staff_pick'][i]) + " " + \
        str(reduced_x_features['backers_count'][i]) + " " + \
        str(reduced_x_features['spotlight'][i]) + " " + \
        str(reduced_x_features['goal'][i]) + " " + \
        str(reduced_x_features['log_goal'][i]) + " " + \
        str(reduced_x_features['log_pledged'][i])  
    kickstarter_X.append(tmp)
    kickstarter_y.append(reduced_y['SuccessfulBool'][i])


max_words = 2000
max_length = 30
vector_length = 16

encoded_docs = [one_hot(d, max_words) for d in kickstarter_X]
padded_docs = pad_sequences(encoded_docs, maxlen=7, padding='post')

X_train, X_test, y_train, y_test = train_test_split(padded_docs, np.array(kickstarter_y)[:, None].astype(int), test_size=0.20, random_state=1234)


model = Sequential()
model.add(layers.Embedding(max_words+1, vector_length, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=7, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# Adding the first CNN layer and Dropout layer
model.add(Dense(128, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))

# Adding a second CNN layer and Dropout layer
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))

# Adding a third CNN layer and Dropout layer
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

# Adding a fourth CNN layer and Dropout layer
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.2))

# For Full connection layer we use dense
# As the output is 1D so we use unit=1
# Adding the output layer
model.add(Dense(1))
# model.add(Dense(1, activation= 'linear'))

print(model.summary())
model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['mse'])

history = model.fit(X_train, y_train, 
        epochs=50, 
        verbose=1,
        validation_data=(X_test, y_test),
        batch_size=256)

scores = model.evaluate(X_test, y_test,
                        verbose=1,
                        batch_size = 256)

dt = RandomForestRegressor(criterion='mse',n_jobs=-1, n_estimators=10, max_depth=6, min_samples_leaf=1, random_state=3)
dt.fit(X_train,y_train)
y_predicted = dt.predict(X_test)
accuracy = dt.score(X_test,y_test)
MSE_score = MSE(y_test,y_predicted)

table = PrettyTable(border=True, header=True, padding_width=1)
table.field_names = ['X', 'y (actual)', 'Predicted']
table.add_row([X_test[15], y_test[15], y_predicted[15]])
table.add_row([X_test[25], y_test[25], y_predicted[25]])
table.add_row([X_test[40], y_test[40], y_predicted[40]])
table.add_row([X_test[47], y_test[47], y_predicted[47]])
table.add_row([X_test[85], y_test[85], y_predicted[85]])
table.add_row([X_test[110], y_test[110], y_predicted[110]])
table.add_row([X_test[202], y_test[202], y_predicted[202]])
table.add_row([X_test[1848], y_test[1848], y_predicted[1848]])
table.add_row([X_test[1857], y_test[1857], y_predicted[1857]])

print(table)