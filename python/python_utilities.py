# INSTALL PACKAGES IN ANACONDA:
# Standard way:
conda install name_of_package
# If not available from current channel:
conda install -c conda-forge name_of_package

# SET GLOBAL WARNINGS IN PYTHON OFF / JUPYTER NOTEBOOK
import warnings
warnings.filterwarnings('ignore')

# SET WORKING DIRECTORY
os.chdir(path_route)

# VISUALIZATION IN JUPYTER NOTEBOOK / IPYTHON
%matplotlib inline
from matplotlib import pyplot as plt

# WATCH ALL COLUMNS IN DATAFRAME / SEE ALL COLUMNS IN DATA FRAME / VIEW ALL COLUMNS IN TABLE
import pandas as pd
from IPython.display import display
pd.options.display.max_columns = None
# OTHER WAY OF ACHIEVING THE SAME
pd.set_option('display.max_columns', len(df.columns)) 

# SEE ALL ELEMENTS IN A FOLDER / LIST ALL FILES IN A DIRECTORY
os.listdir()

# JUPYTER NOTEBOOKS / IPYTHON / HIDE SCROLLING BAR / SCROLL BAR / GRAPHS
# Scroll bar appears when the plot is too BIG. With one click you can hide it or show it completely.
# To eliminate the scrolling bar in outputs, simply click on the left side of the graph over 'Out[number]' part, while it has the scrolling bar. It will
# turn back to normality.
# Stop a Jupyter Notebook in a specific cell after executing whole Notebook:
raise

# CREATION OF A MOCK DATAFRAME
# Option 1: 
x = [1,1000,1001]
y = [200,300,400]
cat = ['first','second','third']
df = pd.DataFrame(dict(speed = x, price = y, place = cat))

# CHECK IF 2 DATA FRAMES ARE EQUAL OR NOT
# First we reset indexes to be safe
df1.reset_index(drop = True, inplace = True)
df2.reset_index(drop = True, inplace = True)
# Option 1
(np.allclose(df1.select_dtypes(exclude=[object]), df2.select_dtypes(exclude=[object]))
   .....:  &
   .....:  df1.select_dtypes(include=[object]).equals(df2.select_dtypes(include=[object]))
   .....: ) 
#Option 2
assert_frame_equal(con_df_filtered, con_df_monthly_filtered, check_dtype = False) 

# FILTER DATA FRAME WITH VALUES IN ANOTHER DATA FRAME
df.loc[df['column_name_to_filter'].isin(reference_values_of_a_column_in_other_df)]

# FREQUENCY TABLES
plms['isfiller'].value_counts()

# DATA QUALITY FUNCTIONS
# 1. Completeness
# PLMS
# First we create the dataframes with the information:
completeness_plms = 100 - plms.isnull().sum()/len(plms)*100
completeness_dqss = 100 - dqss.isnull().sum()/len(dqss)*100
# Convert completeness_plms into data frames:
x = list(completeness_plms.index)
y = completeness_plms
df_completeness_plms = 

# DROP SEVERAL COLUMNS / DELETE SEVERAL COLUMNS
non_predictive = ["NUM_EXPE_VALO",
                  "ID_SOLICITUD_VALO_COMPRO",
                  "DATA_COMUNIC_SOLI",
                  "MOTIU_SOLICITUD",
                  "MODEL_VALO",
                  "OFICINA_COMPETENTE_BE",
                  "PROCEDENCIA_SOL",
                  "REF_CATASTRAL_COMPRO",
                  "ID_AUTOLIQUIDACIO_COMPRO",
                  "COD_INTERNO_BE_CATASTRAL",
                  "DATA_ALTA_REG"]
data = data.drop(non_predictive, 1) 

# READ DATA / READ TABLES / IMPORT DATA / READ CSV FILES / IMPORT CSV
# Import CSV:
df = pd.read_csv(file_dir + file_name, sep = ',', header = 0, encoding = 'latin-1', low_memory = False, )
# Other parameters:
# error_bad_lines=False # to avoid conflicting lines while reading data
# delim_whitespace = True # When the separator separation is tab or tabulator
# header = None # Means NO header
# quoting = 3 # quote = ""
# Use dtypes argument when all rows are consistent in type: i.e. dtype={'user_id': int}
# dtypes in Pandas:
# 'object'
# 'np.int64'
# 'np.float64'
# 'bool'
# 'category'
# dtype doesn't work with dates. For dates we need to use the following:
parse_dates = ['col1date', 'col2date']
df = pd.read_csv(data_dir + df_name, sep = ',', header = 0, encoding = 'latin-1', low_memory = False, dtype={'col1date':'str', 'col2date':'str'}, parse_dates = parse_dates)
# This method automatically parses recognizing the string.
# The dtype argument is highly suggested to be use as it helps in resolving data type conflicts
# The dtype argument can be applied to only 1 or several variables of all the ones available in the dataset
# Import parquet files:
df = pd.read_parquet(file_dir + file_name_perf, engine='pyarrow')

# READ EXCEL / READ EXCEL TABLES / READ .XLS / READ XLS
df = pd.read_excel('name_of_file.xlsx', converters={'column_1': np.int32, 'column_n':  np.int32}) 

# MERGE / JOIN DATA FRAMES
df = pd.merge(df1, df2, on = ['variable_1', 'variable_2'], how = 'inner')
# how can be changed to 'outer', 'left' or right

# DATA TYPE CONVERSION / CHANGE DATA TYPE / CONVERT DATA TYPE
# Data Type Conversions:
# -> String:
df['id']= df['id'].astype('str')
to_string = ['col_1', 'col_2', 'col_3', ..., 'col_n']
df[to_string] = df[to_string].apply(lambda x: x.astype(str))
# -> String ALL COLUMNS / Convert all columns to string
df = df.astype(str)
# -> Category:
plms[to_cat] = plms[to_cat].apply(lambda x: x.astype('category'))
df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))
# -> Numeric:
df['column_to_numeric'] = pd.to_numeric(df.column_to_numeric, errors = 'coerce')
# -> Integer
df['column_to_int'].astype(np.int64)
# Convert all columns to numeric type:
df = df.apply(pd.to_numeric, errors='coerce')

# MISSINGS / DATA QUALITY ASSESSMENT / DATA QUALITY
# Count number of missings in entire dataframe:
sum(tl_view.isnull().sum())
# Show number of missings per column in percentage:
df.isnull().sum()/len(df)*100
# Eliminate missings from entire dataframe:
df.dropna(axis=0, how='any', inplace = True) # Eliminate rows.
df.dropna(inplace=True)
# Total number of missings in entire dataset
df.isnull().sum()
# Count number of missings for a column:
df['hardship_amount'].isnull().sum()
# Show number of missings for each column:
df.isnull().sum()
# Show number of missings per column in percentage:
df.isnull().sum()/len(df)*100
# Replace missing values with 0:
df[np.isnan(df)] = 0
# Replace all blank spaces with Nan:
df.replace(r'\s+', np.nan, regex = True, inplace = True)
# Show table or registries without missing values (complete table sample):
df[~df.isnull().any(axis=1)]

# SORT VALUES / SORT A LIST / SORT A DATAFRAME / ORDER BYR VALUE
df.sort_values(by = ['col1'], ascending = True, inplace = True) # 'ascending = False' for the other result

# DICTIONARIES
# Create an initial dictionary with specific size:
d = {}
for i in range(4000000):
    d[i] = None
# Change name of a key inside dictionary:
dict[name_of_new_key] = dict.pop(name_of_old_key)
# Print elements of a dictionary
list(name_of_dict)
name_of_dict.keys()
list(name_of_dict.keys())
# Access elments of the dictionary:
list(name_of_dict)[position_wanted]

# LISTS
# Append a list in a for loop:
a=[]
for i in range(5):    
    a.append(i)
a # the list with the new items.
# Convert a list to a Data Frame / Convert a list to a dataframe
# Other form of creating a data frame: 
x = [1,1000,1001]
y = [200,300,400]
df = pd.DataFrame(dict(name_of_column_1_without_quotes = x, name_of_column_2_without_quotes = y))

# ELEMENTS OF A LIST INSIDE ANOTHER LIST
list(set(list1) - set(list2))

# ELEMENTS OF A DATA FRAME INSIDE ANOTHER DATA FRAME
.isin()

# SORT / ORDER
sorted(list)
list.sorted()

# SEQUENCES / RANGES
# Create a simple sequence in Pandas:
[i for i in range(beginning, ending + 1)] # Range ends below the upper limit by 1 position

# SELECT / FILTER DATA / FILTER A DATASET / FILTER DATAFRAME / EXCLUDE / NOT SELECT / ASSIGNATION / ASSIGN
# Index numeric, column by name:
df.loc[df.index[number_desired_of_index], 'name_of_column']
# Select all dataframe except for one column
df.loc[:, df.columns != 'name_of_column_to_exclude']
# Obtain all values not included in a list of values: list_with_values_of_variable_to_filter
dfx = df[~df['name_of_variable'].isin(list_with_values_of_variable_to_filter)] # You obtain the registries of the dataframe that do not have the values of the list in the specified variable
dfx = df[df['name_of_variable'].isin(list_with_values_of_variable_to_filter)] # You obtain the registries of the dataframe that DO have the values of the list in the specified variable
unified['col-2'].isin(eventcodesprod) #Condition: 'take registries where 'col-2' has any value inside eventcodesprod

# CREATE NEW COLUMN / ADD NEW COLUMN / ADD NEW FEATURE / ADD NEW VARIABLE
df['name_of_new_column'] = pd.Series(np.nan , index = df.index)

# RENAME A SINGLE COLUMN / RE-NAME A COLUMN / CHANGE NAME OF A COLUMN / CHANGE COLUMN NAME
df.rename(columns={'column_name_to_change':'new_column_name'}, inplace=True)

# SAVE A PICKLE FILE / PYTHON´s .RDATA
# Save a model:
import pickle
pickle.dump(name_of_object_to_save, open('route/desired_name_of_file.pickle', 'wb'))
# Load model:
loaded_model = pickle.load(open('route/desired_name_of_file.pickle', 'rb'))
result = loaded_model.score(X_test, Y_test) *# for example, to obtain predictions
print(result)
# Method useful only for pandas dataframes:
df.to_pickle('name_of_file_saved.pickle')  # where to save it, usually as a .pkl
object_saved = pd.read_pickle('name_of_file_saved.pickle')

# CROSS VALIDATION / TRAIN - TEST SPLITTING / DIVIDE DATASET / SELECT TRAIN AND TEST
train, test = train_test_split(df, test_size = perecntage_of_test_subset_in_decimal)

# NUMPY ARRAY CONVERSION TO DATA FRAME
new_df = pd.DataFrame(data = name_of_numpy_array[0: , 0:],    # all the matrix are the values of the new dataframe
                      index = range(0,len(name_of_numpy_array)),    # index goes from 0 until the end of the dataframe
                      columns = [('V' + '_' + str(i)) for i in range(0,(name_of_numpy_array.shape[1]))])  # columns are generic as V_1, V_2, ... V_n

# CBIND IN PANDAS / COMBINE COLUMNS / PASTE COLUMNS
first_df.reset_index(drop = True, inplace = True)
second_df.reset_index(drop = True, inplace = True)
united_df = pd.concat([first_df, second_df], axis=1)
united_df.reset_index(drop = True, inplace = True)

# RBIND IN PANDAS / COMBINE ROWS / PASTE ROWS
first_df.reset_index(drop = True, inplace = True)
second_df.reset_index(drop = True, inplace = True)
united_df = pd.concat([first_df, second_df], axis=0)
united_df.reset_index(drop = True, inplace = True)

# RESET INDEX OF DATAFRAME
df.reset_index(drop = True, inplace = True)

# COLUMN NAMES REPLACE STRINGS / SUBSTITUTE VALUES IN COLUMN NAMES / COLNAMES
# Change points in column names for underscore (_):
df.columns=df.columns.str.replace('.','_')
# To remove white spaces everywhere for nothing:
df.columns = df.columns.str.replace(' ', '')
# To replace white space everywhere for underscore (_):
df.columns = df.columns.str.replace(' ', '_')
# To remove white space at the beginning of string:
df.columns = df.columns.str.lstrip()
# To remove white space at the end of string:
df.columns = df.columns.str.rstrip()
# To remove white space at both ends:
df.columns = df.columns.str.strip()
# To replace white space at the beginning:
df.columns = df.columns.str.replace('^ +', '_')
# To replace white space at the end:
df.columns = df.columns.str.replace(' +$', '_')
# To replace white space at both ends:
df.columns = df.columns.str.replace('^ +| +$', '_')

# DATE / DATETIME / TIMESTAMP
# Convert String to Date Time / String -> Datetime / Change a string type: 20180931
df['name_of_column'] = df['name_of_column'].apply(pd.to_datetime, format = '%Y%m%d', errors = 'coerce')  
# Convert object to datetime/date type:
df['name_of_column'] =  pd.to_datetime(df['name_of_column'], format='%d%b%Y:%H:%M:%S.%f')
# Convert a string variable a dataframe to date.
# %Y is 2018
# %m is 09
# %d is 31
# CHANGE DATE (DATETIME64NS) FROM ONE FORMAT TO ANOTHER / FORMATS
df['datetime64ns_column_to_change_format'] = pd.to_datetime(df.datetime64ns_column_to_change_format)
df['datetime64ns_column_to_change_format'] = df['datetime64ns_column_to_change_format'].dt.strftime('format_desired')
# Example of formats:
'%Y-%m-%d %H:%M:%S'
# Date difference / Difference of Dates in years
df.diff_dates_col = df.date1 - df.date2
df.diff_dates_col / dt.timedelta(days = 365)
# Date difference / Difference of Dates in days
df.diff_dates_col = df.date1 - df.date2
df.diff_dates_col / dt.timedelta(days = 1)

# SUBSTRING / CHARACTER SELECTION / STRING SELECTION
df.name_of_column = df.name_of_column.str.slice(0, 9) # Select characters from 0 to 8 (9-1)
df['name_of_column'] = df['name_of_column'].str.slice(0, 9) # Select characters from 0 to 8 (9-1)
df.name_of_column = df.name_of_column.str[:9]
df['name_of_column'] = df['name_of_column'].str.replace("string_pattern_to_be_replaced", 'string_pattern_to_substitute')

# GROUP BY
# Group by and Sum
df.groupby(['ke1', 'key2'])['col_to_operate_aggregate'].agg('sum')

# PLOT
# The arguments for Seaborn work as well for Matplotlib.

# COMPLETE EXAMPLE
#------Graph Size------
#Plot original time series and daily mean
# Size of the graph 
fig = plt.figure(figsize=(28, 18))
# Size of the graph in a more complete and complex way (more features)
fig, ax = plt.subplots(figsize=(28, 18))
# Adding a subgraph with a 2nd Time Series in the same plot
ax = fig.add_subplot(1,1,1)
#------Ticks Size------
# Size of Ticks, How big are the numbers and marks in each axis?
plt.tick_params(axis = 'both', which = 'major', labelsize = 24)
plt.tick_params(axis = 'both', which = 'major', labelsize = 24)
#------Ticks Frequency / Ticks Density------
# X Axis
ax.xaxis.set_major_locator(ticker.MultipleLocator(250)) # The number 250 represents the desired increase pattern in the axis
ax.xaxis.set_major_formatter(ticker.ScalarFormatter()) # X Axis
# Y Axis
ax.yaxis.set_major_locator(ticker.MultipleLocator(250)) # The number 250 represents the desired increase pattern in the axis
ax.yaxis.set_major_formatter(ticker.ScalarFormatter()) # y Axis
#------Names of Labels and Size of Labels------
# Title and Size of Labels in each Axis
plt.title("Hola", size = 25) # Main Title
plt.xlabel('Time difference (Days)', size = 22) # X Axis Title
plt.ylabel('Count of Occurrences', size = 22) # Y Axis Title
# Other way of achieving the same using the parameter ax:
ax.set_title('Time Series Inventory Level', fontsize = 24) # Main Title
ax.set_xlabel('Time (Daily Aggregation)') # X Axis Title
ax.set_ylabel('Inventory Level (Number of Units in Stock)') # Y Axis Title
#------Legend------
# Use of argument "label = "
dayly_mean.plot(ax=ax, color='b', label = 'Time Series of Inventory Levels on Daily Data')
monthly_mean.plot(ax=ax, color='r', label = 'Time Series of Inventory Levels on Monthly Data')
# Add legend ('bbox_to_anchor' is the position of the legend; 'size' is size of letters in legend)
plt.legend(bbox_to_anchor = (0.02, 0.97), loc=2, borderaxespad=0., prop={'size': 20})

# SEABORN
# Histogram without density
import seaborn as sns
sns.set_style("darkgrid") # Setting the style of the background
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k') # Set size of graph plot
ax = plt.axes() # Creating the axes to edit properties
fig = sns.distplot(part1a_nonull.time_diff, kde = False, bins = 100, ax = ax) # Assigning object to the graph
plt.title("Histogram of parts that changes from NO Delivery Delay to Availability Delay'") # Assigning title to the plot
plt.xlabel("Days of Delay") # Assigning title to the x label
plt.ylabel("Count of Occurrences") # Assigning title to the y label
plt.show(fig)

# MATPLOTLIB
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(35, 28)) # Set size of the graph
plt.rcParams.update({'font.size': 22})
plt.hist(pd.to_numeric(tpa3cf_final['distance']), bins='auto')  # arguments are passed to np.histogram
ax.tick_params(direction='out', length=6, width=2, colors='black',grid_color='r', grid_alpha=0.5) # change size of axis with pyplot from matplotlib NOT seaborn
plt.title("Defects distribution for TP A3 CP")
plt.xlabel("Distance Days") # Assign name of the x label
plt.ylabel("Frequency") # Assign name of the y label
plt.title("Distribution of Defects based on Days Distance to the Event") # Assign title of the graph
plt.show()
# 2 Histograms overposed
import numpy as np
import pylab as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(22, 12), dpi=80, facecolor='w', edgecolor='k') # Size of graph
# 1 1
plt.hist(part2b_nonull.loc[part2b_nonull.time_diff <100, 'time_diff'], edgecolor=(0.1,0,0,1), lw=1, facecolor='red', rwidth=0.8, bins = 50, label = "Change from Delivery Delay to Availability Delay type 'OUTLIER'") # First hist
# 1 0
plt.hist(part2a_nonull.loc[part2a_nonull.time_diff < 100, 'time_diff'], edgecolor=(0.1,0,0,1), lw=1, facecolor='aqua', rwidth=0.8, bins = 50, label = "Change from Delivery Delay to Availability Delay type 'EXTREME'") # Second hist
plt.xlabel('Delay Days', size = 19) # Size of x label and name
plt.ylabel('Count of Occurrences', size = 19) # size of y label and name
plt.tick_params(axis='both', which='major', labelsize=18) # Size of axis x and y numbers or ticks 
axvlines(part2b_nonull.median(), color='red', label = 'Median - Change from Deliv. Delay to Avail. Delay type <<OUTLIER>>', linewidth=4) # Vertical line (see functions)
axvlines(part2a_nonull.median(), color='aqua', label = 'Median - Change from Deliv. Delay to Avail. Delay type <<EXTREME>>', linewidth=4) # Second Vertical line (see functions)
plt.legend(bbox_to_anchor = (0.16, 0.97), loc=2, borderaxespad=0., prop={'size': 15}) # Legend added automatically with labels names
# bbox_to_anchor states position of the legend box in format (x,y)
# loc = 2 states above the graph
# prop = {'size': 15} indicates size of letters and icons in legend

# How to edit x or y axis ticks density?
mport matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn.apionly as sns
import numpy as np
sns.set_style("darkgrid") # Setting the style of the background
figure(num=None, figsize=(15, 7), dpi=80, facecolor='w', edgecolor='k') # Set size of graph plot
ax = plt.axes() # Creating the axes to edit properties
# fig = sns.distplot(qtip_results.loc[(qtip_results.time_diff < 300) , 'time_diff'], kde = False)
fig = sns.barplot(x = df_grouped.loc[df_grouped.time_diff <100 , 'time_diff'] , y = df_grouped.cnt_of_subsequent_so)
plt.title("Time Difference for parts that change from Qu-TIP TI to Stock Out") # Assigning title to the plot
plt.xlabel("Days Between QuTI-P and Stock Out") # Assigning title to the x label
plt.ylabel("Cases count") # Assigning title to the y label
ax.yaxis.set_major_locator(ticker.MultipleLocator(250))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.show(fig)

# How to edit x or y axis density but when they are categorical and you want to set how many ticks there are?
ax.locator_params(nbins=8, axis='x')

# VIEW / VIEW A TABLE / VIEW A DATA FRAME / SEE A TABLE / SEE A DATA FRAME / VIEW ALL COLUMNS IN A DATAFRAME DATA FRAME DATA SET / IPYTHON JUPYTER
# View all columns in table:
import pandas as pd
from IPython.display import display
pd.options.display.max_columns = None

# RESET INDEX OF A DATAFRAME
df.reset_index(drop = True, inplace = True)

# APPLY / LAMBDA FUNCTION 
df[columns_to_apply_lambda] = df[columns_to_apply_lambda].apply(lambda x: write_function_as_it_was_our_variable_and_then_change_the_column_by_x)
# Where x in our mind would be where we put the column.
# The trick is to write the function as it was with our variable, and then changing the column by 'x'
# Example:
plms[to_cat] = plms[to_cat].apply(lambda x: x.astype('category'))

# PRINT CSV TABLE / EXPORT CSV TABLE / PRINT TEXT FILES / EXPORT
df.to_csv(directory_with_slash_at_end + name_of_file_with_extension, sep='|', index = False)
