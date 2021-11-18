import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 67)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale = 1)
#from matplotlib import inline
plt.style.use('ggplot')

np.random.seed(7)
from prettytable import PrettyTable
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE

from keras.models import Sequential
from keras import layers
from keras.layers import LSTM,Dropout
from keras.layers import Dense,Conv1D,MaxPooling1D
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

#CODE BLOCK 1 data wrangling. 

kickstarter = pd.read_csv('../../kickstarter_data_full.csv', index_col=0)
print(kickstarter.columns)
cols_to_drop = ['friends', 
                'is_starred',
                'is_backing',
                'permissions',
                'profile',
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
                'deadline_month',
                'deadline_day',
                'deadline_yr',
                'deadline_hr',
                'state_changed_at_month',
                'state_changed_at_day',
                'state_changed_at_yr',
                'state_changed_at_hr',
                'created_at_month',
                'created_at_day',
                'created_at_yr',
                'created_at_hr',
                'launched_at_month',
                'launched_at_day',
                'launched_at_yr',
                'launched_at_hr'
                ]
kickstarter.drop(labels=cols_to_drop, axis=1, inplace=True)

print(kickstarter.columns)

kickstarter['disable_communication'] = kickstarter['disable_communication'] * 1 #converts type bool to 0 for false and 1 for true
kickstarter['staff_pick'] = kickstarter['staff_pick'] * 1 #converts type bool to 0 for false and 1 for true
kickstarter['spotlight'] = kickstarter['spotlight'] * 1 #converts type bool to 0 for false and 1 for true
print(kickstarter.head())