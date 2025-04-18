# OBTAIN HEAD IN GIT REPO AUTOMATICALLY / PROJECT STRUCTURE / SOURCE FOLDER / DIRECTORY / MASTER / AVOID HARDCODING
import subprocess
import os
from IPython.display import Image, display

# Get the absolute path to the Git repo root (HEAD)
def get_git_root():
    return subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    ).stdout.strip()

name_file = "comparison.PNG"  # This is an example of a file located in 'images' folder
# Build the full image path
repo_root = get_git_root()
image_path = os.path.join(repo_root, "images", name_file)  # 'images' is the folder inside the HEAD git repo folder
# Display if found
display(Image(filename=image_path))

# RUN JUPYTER NOTEBOOK OUTSIDE JUPYTER NOTEBOOK
from utils import repo_root  # You need to create a new file named utils.py at same level as your current notebook ATTENTION! Name is utils.py not utils.ipynb IT IS A PYTHON FLAT FILE
from IPython.display import Image, display
import os

# Negation in Python is: ~ instead of: !
# We must always convert all columns with its respective types. "Repass" the correct types. joblib y groupby require it.

# HEAD OF SCRIPTS /
############################################################################
# Title: XXX
# Date: XX of name_of_month of XXXX
############################################################################

# SEPARATOR OF CHUNK OF CODES
############################################################################
# NAME OF A PART
############################################################################

############################################################################
                         # NAME OF A PART
############################################################################

# FIND COLUMN NAMES THAT CONTAIN SPECIFIC STRING
[col for col in df.columns if '_y' in col]  # Searching for column names with "_x" in name (as if they were duplicated).

# GUARANTEE TRAIN AND TEST SETS HAVE VARIABLES / FEATURES IN THE SAME ORDER
X_test = X_test.reindex(columns = X_train.columns, fill_value=0)

# XLRD:
# xlrd should be version 1.2.0 when installing with pip command.

# VENV VIRTUAL ENVIRONMENT SOURCE PYTHON VIRTUAL ENVIRONMENT ACTIVATE
source C:/Projects/2b9d13-accelerate-time-to-value-model/venv/Scripts/activate

# POWERSHELL ACTIVATION FOR VENV IN PYCHARM
# It requires opening before Powershell in Admin Mode
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
# It requires after pressing option "A" ("Yes to All")

# SEE ALL COLUMNS:
pd.options.display.max_columns = None

# SEE ALL ROWS:
pd.options.display.max_rows = None

# PIP INSTALL REQUIREMENTS
pip install -r requirements.txt

# DELETE VARIABLES / CLEAR WORKSPACE / CLEAR WORK SPACE 
 %reset -f 

# MISSING SELECTION / NP.NAN USE
# You don't use np.nan when you want to select missing data, you use:
df.column_name.isnull() # This create a conditional boolean vector to put in the rows position and select data that satisfy true
df.column_name.notnull() # This create a conditional boolean vector to put in the rows position and select data that satisfy true

# OBTAIN LEVELS OR VALUES OF A STRING OR OBJECT OR CATEGORICAL VARIABLE
df.column_name.unique()

# OBTAIN LEVELS FROM CATEGORICAL / OBJECT / STRING VARIABLE:
def levels(var):
  return sorted([x for x in list(var.unique()) if str(x) != 'nan'])

# DROP INFINITE VALUES - REPLACE INFINITE BY MISSINGS - INFINITY 
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# RUN PYTHON SCRIPT INSIDE ANOTHER PYTHON SCRIPT
exec(open("C:/ANIBAL/FORMATION/PERSONAL/2_autos_uciml/scripts/1_libraries_autos.py").read())

# OBTAIN COLUMN TYPES IN ALPHABETICAL ORDER / PRINT COLUMN TYPES ALPHABETICALLY
print(df.sort_index(axis=1).dtypes)
df.dtypes.sort_index()
df.sort_index(axis=1).dtypes # Does the same

# OPEN JUPYTER NOTEBOOK FROM CMD OR COMMAND LINE
# 1. Go to windows explorer destination where your head directory of the project is
# 2. Erase what appears on the horizontal bar of the location route at the top, and write "cmd" without quotation marks, then press enter.
# 3. Write "python -m notebook" without quotation marks in the terminal, and then press enter.
python -m notebook

# MEASURE TIME IN A PYTHON PROCEDURE
import time
start = time.time()
-all the procedure you want to do in python-
end = time.time()
print("Time taken was {} seconds").format(end - start)

# SHOW COMPLETE SIZE OF LIST / LIST SIZE / INCREASE LIST SIZE / SHOW MAXIMUM ROWS
pd.options.display.max_rows = None

# SHOW COMPLETE TABLE / SHOW ALL COLUMNS / VIEW ALL COLUMNS / SHOW MAXIMUM COLUMNS
pd.options.display.max_columns = None 

# ASSIGNMENT TO A COLUMN DEPENDING ON VALUES OF OTHER COLUMN / CONDITIONAL ASIGNMENT TO A COLUMN
    conditions = [
            (testdf['LEVEL_L1 Small Size Projects'] == 0) & (testdf['LEVEL_L2 Medium Size Projects'] == 0) & (testdf['LEVEL_L3 Large Size Projects'] == 0) & (testdf['LEVEL_L3+ Very Large Size Projects'] == 0), 
            (testdf['LEVEL_L1 Small Size Projects'] == 1),
            (testdf['LEVEL_L2 Medium Size Projects'] == 1) & (testdf['TOTAL_VALUE_KEURO'] < 2000),
            (testdf['LEVEL_L2 Medium Size Projects'] == 1) & (testdf['TOTAL_VALUE_KEURO'] >= 2000),
            (testdf['LEVEL_L3 Large Size Projects'] == 1),
            (testdf['LEVEL_L3+ Very Large Size Projects'] == 1)
            ]
    choices = [
            testdf['TOTAL_VALUE_KEURO'] * 0,
            testdf['TOTAL_VALUE_KEURO'] * 0.015,
            testdf['TOTAL_VALUE_KEURO'] * 0.01,
            testdf['TOTAL_VALUE_KEURO'] * 0.005,
            testdf['TOTAL_VALUE_KEURO'] * 0.005,
            testdf['TOTAL_VALUE_KEURO'] * 0.005            
            ]
    testdf['BUDGET_COST'] = np.select(conditions, choices, default = 0)
    
# DTYPES ALPHABETICAL / PRINT DATAFRAME WITH COLUMNS IN ALPHABETICAL ORDER / VIEW DATAFRAME WITH COLUMNS ORDERED ALPHABETICALLY / ORDER COLUMNS OF DATA FRAME ALPHABETICALLY
df.sort_index(axis=1, inplace=True)

# READ MULTIPLE FILES IN PANDAS AT THE SAME TIME
import pandas as pd
import glob
path = r'C:\DRO\DCL_rawdata_files' # use your path
all_files = glob.glob(path + "/*.csv")
li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
frame = pd.concat(li, axis=0, ignore_index=True)

# FREQUENCY TABLES
def freqt(var):
    return print(var.value_counts(dropna = False))

# PERCENTAGE OF MISSINGS
def pmiss(df):
    return df.isnull().sum() * 100 / len(df)

# ASSIGNATION DEPENDING ON VALUES OF OTHER DATA FRAME / MAP
dictionary_with_new_values = dict({
    'old_value1': 1,
    'old_value2': 2,
    'old_value3': 3,
    'old_value4': 4
})
df.replace({"column_to_replace_values": dictionary_with_new_values})
# Other way of doing it:
df_currencies['Month'] = df_currencies['Month'].map(month_dict)
# REMEMBER THAT THE NEW VALUES SHOULD BE OF THE SAME TYPE AS THE PREVIOUS COLUMN!!!
# IF NOT, YOU SHOULD CREATE A NEW COLUMN
# If the values that we should map are in 2 columns of a 3rd or other data frame. It should be with:
df1['column_to_change'] = df1['column_to_change'].map(df_mapping.set_index('column_with_same_name_as_column_to_change')['column_with_new_values'])

# CHANGE COLOR OF JUPYTER CELLS
# <font color = "blue"> 7. Assessment of Quality of the Missing Exclussion: <font>
<font color = "blue"> 7. Assessment of Quality of the Missing Exclussion: <font>

# INSTALL PACKAGES IN ANACONDA:
# Standard way:
conda install name_of_package
# If not available from current channel:
conda install -c conda-forge name_of_package

# READ SEVERAL FILES IN AN ADRESS
path = r"/Users/anibalmartinez-sistac/Documents/anibal/master_ai/descubrimiento_de_informacion_en_textos/desc_info_textos_tarea_2/data/fc2/" # SPECIFY YOUR PATH
files = [file for file in os.listdir(path) if not file.startswith('.')]

# SET GLOBAL WARNINGS IN PYTHON OFF / JUPYTER NOTEBOOK
import warnings
warnings.filterwarnings('ignore')

# SET WORKING DIRECTORY
os.chdir(path_route)

# VISUALIZATION IN JUPYTER NOTEBOOK / IPYTHON
%matplotlib inline
from matplotlib import pyplot as plt

# WATCH ALL COLUMNS IN DATAFRAME / SEE ALL COLUMNS IN DATA FRAME / VIEW ALL COLUMNS IN TABLE / JUPYTER WATCH ALL COLUMNS
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

# CREATION OF A MOCK DATAFRAME / CREATION OF A NEW DATA FRAME / CREATE A NEW DATA FRAME
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
import collections, numpy
collections.Counter(numpy_array) # Being numpy_array a Numpy array

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

# READ DATA / READ TABLES / IMPORT DATA / READ CSV FILES / IMPORT CSV / LOAD DATA/ LOADING TABLES 
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

# DATA TYPE CONVERSION / CHANGE DATA TYPE / CONVERT DATA TYPE / DATA TYPE CASTING
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
# -> NUMERIC:
df['column_to_numeric'] = pd.to_numeric(df.column_to_numeric, errors = 'coerce')
# Several columns to numeric type:
df[cols].apply(pd.to_numeric, errors='coerce')
# Convert all columns to numeric type:
df = df.apply(pd.to_numeric, errors='coerce')
# -> INTEGER
df['column_to_int'].astype(np.int64)

# MISSINGS / DATA QUALITY ASSESSMENT / DATA QUALITY
# Count number of missings in entire dataframe:
sum(tl_view.isnull().sum())
# Show number of missings per column in percentage:
df.isnull().sum()/len(df)*100
# Eliminate missings from entire dataframe:
df.dropna(axis=0, how='any', inplace = True) # Eliminate rows.
df.dropna(inplace=True) # The same in another way
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
df_revisions = df_revisions.replace(r'^\s+$', np.nan, regex=True)
df.replace(r'\s+', np.nan, regex = True, inplace = True) # Alternative solution that may no work
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
# Create a dictionary with keys of the dictionary the values of a list, and values of the dictionary empty lists
dic = dict.fromkeys(list(list_with_names), [])

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

# SELECT DATA FROM ONE TYPE / SUBSET DATA
# Select numerical data (it includes integer and float64 as well):
df._get_numeric_data() # 2nd option
# Select float64 data:
df.select_dtypes(include=['float64'])
# Select integer data:
df.select_dtypes(include=['integer'])
# Select object/string data
df.select_dtypes(include=['object'])
# Select categorical Data:
df.select_dtypes(include=['category'])
# Select datetime / date:
df.select_dtypes(include=['datetime64'])

# SELECT / FILTER DATA / FILTER A DATASET / FILTER DATAFRAME / EXCLUDE / NOT SELECT / ASSIGNATION / ASSIGN
# Index numeric, column by name:
df.loc[df.index[number_desired_of_index], 'name_of_column']
# Select all dataframe except for one column
df.loc[:, df.columns != 'name_of_column_to_exclude']
# Obtain all values not included in a list of values: list_with_values_of_variable_to_filter
dfx = df[~df['name_of_variable'].isin(list_with_values_of_variable_to_filter)] # You obtain the registries of the dataframe that do not have the values of the list in the specified variable
dfx = df[df['name_of_variable'].isin(list_with_values_of_variable_to_filter)] # You obtain the registries of the dataframe that DO have the values of the list in the specified variable
unified['col-2'].isin(eventcodesprod) #Condition: 'take registries where 'col-2' has any value inside eventcodesprod
# Filter data frame when column must be equal to several values, without repeating logical operators, only specifying different values:
df.loc[ (df.column_to_filter.isin(["Value_desired_1", "Value_desired_2"])) , : ]

# CREATE NEW COLUMN / ADD NEW COLUMN / ADD NEW FEATURE / ADD NEW VARIABLE
df['name_of_new_column'] = pd.Series(np.nan , index = df.index)

# RENAME A SINGLE COLUMN / RE-NAME A COLUMN / CHANGE NAME OF A COLUMN / CHANGE COLUMN NAME
df.rename(columns={'column_name_to_change':'new_column_name'}, inplace=True)

# SAVE A PICKLE FILE / PYTHON´s .RDATA
# Save a model:
import pickle
pickle.dump(name_of_object_to_save, open('route/desired_name_of_file.pickle', 'wb'))
# Method useful for pandas dataframes:
df.to_pickle('name_of_file_saved.pickle')  # where to save it, usually as a .pkl
# Way of saving several objects as pickle file:
tuple_of_objects = (object1, object2, object3, object4)
filename = "C:/ANIBAL/PROJECTS/3_CLTV/Processing%20CLTV/output/best_model.pickle"  # A specific route inside your computer
with open(filename, 'wb') as f:
    pickle.dump(tuple_of_objects, f)

# LOAD PICKLE FILE
# Load model:
loaded_model = pickle.load(open('route/desired_name_of_file.pickle', 'rb'))
result = loaded_model.score(X_test, Y_test) *# for example, to obtain predictions
print(result)
# For Data Frames:
object_saved = pd.read_pickle('name_of_file_saved.pickle')
# Way of loading several different objects:
filename = "C:/ANIBAL/PROJECTS/3_CLTV/Processing%20CLTV/output/best_model.pickle"  # A specific route inside your computer
with open(filename, "rb") as f:
    saved_objects = pickle.load(f)
    
# SAVE PYTHON SESSION / SAVE WORKSPACE / SAVE WORK SPACE / PYTHON .RDATA
# Import the needed library "shelve"
import shelve
# Declare the constants (for example directory route where we are saving the session file):
filename='C:/ANIBAL/PROJECTS/3_CLTV/Processing%20CLTV/output/session_saved.dat'
# Execute all following lines in one chunk to save all objects:
my_shelf = shelve.open(filename,'n') # 'n' for new
for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()

# OPEN PYTHON SESSION / SAVE WORKSPACE / SAVE WORK SPACE / PYTHON .RDATA
# Import the needed library "shelve"
import shelve
# Declare the constants (for example directory route where we are saving the session file):
filename='C:/ANIBAL/PROJECTS/3_CLTV/Processing%20CLTV/output/session_saved.dat'
# Execute all following lines in one chunk to open all objects:
my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()

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
# Convert column names to lower case:
df.columns = map(str.lower, df.columns)
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

# REGEX
# Eliminate all blank spaces in column values before the beginning of the first letter and after the end of the last letter you see in the PC:
df.column_desired = df.column_desired.str.strip()
# Elimination of contiguous unnecesary blank spaces:
df = df.applymap(lambda x: np.nan if isinstance(x, str) and (not x or x.isspace()) else x)

# DUMMIES / ONE-HOT ENCODING / ONE HOT ENCODING / BINARIZING DATA
df = pd.get_dummies(df, drop_first = True)
# another form of doing it:
values = df.values # Convert rows of df in elements of an array
encoder = LabelEncoder() # Create an object of the LabelEncoder() function
values[:,4] = encoder.fit_transform(values[:,4]) # Convert the 4th column to numeric (starting from 0, so it would be the 5th: 0, 1, 2, 3, 4).
values = values.astype('float32') # We ensure all the data is float after conversion
scaler = MinMaxScaler(feature_range=(0, 1)) # Normalize features with a MinMaxScaler object
scaled = scaler.fit_transform(values) # Apply normalization to our data

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
df['datetime64ns_column_to_change_format'] = pd.to_datetime(df.datetime64ns_column_to_change_format) # 1st step of 2
df['datetime64ns_column_to_change_format'] = df['datetime64ns_column_to_change_format'].dt.strftime('format_desired') # 2nd step of 2
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
df.groupby(['ke1', 'key2'], as_index=False)['col_to_operate_aggregate'].agg('sum')
df.sort_values(['column_that_we_want_the_last']).groupby('column_identifier', as_index=False).last()

# PLOT
# The arguments for Seaborn work as well for Matplotlib.

# SMOTE / UNBALANCED CLASSES
from imblearn.over_sampling import SMOTE
smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(X_trainmodels1, y_trainmodels1)
rfmodels1.fit(X_sm, y_sm)
y_predmodels1 = rfmodels1.predict(X_testmodels1)
y_truemodels1 = y_testmodels1
y_scoresmodels1 = rfmodels1.predict_proba(X_testmodels1)[:,1]
accuracy_score(y_testmodels1, y_predmodels1)

# COMPLETE EXAMPLE
import matplotlib.ticker as ticker
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
plt.tick_params(axis = 'x', which = 'major', labelsize = 24)
plt.tick_params(axis = 'y', which = 'major', labelsize = 24)
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
# Rotate x axis / Rotate 90 degrees x axis / Vertical axis labels:
plt.xticks(rotation='vertical') # Best way
ax.set_xticklabels(df.index, rotation=90) # Alternative way

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

# CROSS VALIDATION / TRAIN TEST SPLITTING / CROSS-VALIDATION / 
labels = finaldf.SUCC
X_train, X_test, y_train, y_test = train_test_split(finaldf, labels, test_size = 0.3)

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

# ADDING VERTICAL LINES TO PLOTS
# ys is the value where we want the line to be marked.
def axhlines(ys, ax=None, **plot_kwargs):
    """
    Draw horizontal lines across plot
    :param ys: A scalar, list, or 1D array of vertical offsets
    :param ax: The axis (or none to use gca)
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    if ax is None:
        ax = plt.gca()
    ys = np.array((ys, ) if np.isscalar(ys) else ys, copy=False)
    lims = ax.get_xlim()
    y_points = np.repeat(ys[:, None], repeats=3, axis=1).flatten()
    x_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(ys), axis=0).flatten()
    plot = ax.plot(x_points, y_points, scalex = False, **plot_kwargs)
    return plot

# ADDING HORIZONTAL LINES TO PLOTS
# xs is the value where we want the line to be marked.
def axvlines(xs, ax=None, **plot_kwargs):
    """
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param ax: The axis (or none to use gca)
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    if ax is None:
        ax = plt.gca()
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    lims = ax.get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(xs), axis=0).flatten()
    plot = ax.plot(x_points, y_points, scaley = False, **plot_kwargs)
    return plot
def scatter_matrix_all(frame, alpha=0.5, figsize=None, grid=False, diagonal='hist', marker='.', density_kwds=None, hist_kwds=None, range_padding=0.05, **kwds):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.artist import setp
    import pandas.core.common as com
    from pandas.compat import range, lrange, lmap, map, zip
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    df = frame
    num_cols = frame._get_numeric_data().columns.values
    n = df.columns.size
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=figsize, squeeze=False)

    # no gaps between subplots
    fig.subplots_adjust(wspace=0, hspace=0)

    mask = com.notnull(df)
    marker = _get_marker_compat(marker)

    hist_kwds = hist_kwds or {}
    density_kwds = density_kwds or {}

    # workaround because `c='b'` is hardcoded in matplotlibs scatter method
    kwds.setdefault('c', plt.rcParams['patch.facecolor'])

    boundaries_list = []
    for a in df.columns:
        if a in num_cols:
            values = df[a].values[mask[a].values]
        else:
            values = df[a].value_counts()
        rmin_, rmax_ = np.min(values), np.max(values)
        rdelta_ext = (rmax_ - rmin_) * range_padding / 2.
        boundaries_list.append((rmin_ - rdelta_ext, rmax_+ rdelta_ext))

    for i, a in zip(lrange(n), df.columns):
        for j, b in zip(lrange(n), df.columns):
            ax = axes[i, j]

            if i == j:
                if a in num_cols:    # numerical variable
                    values = df[a].values[mask[a].values]
                    # Deal with the diagonal by drawing a histogram there.
                    if diagonal == 'hist':
                        ax.hist(values, **hist_kwds)
                    elif diagonal in ('kde', 'density'):
                        from scipy.stats import gaussian_kde
                        y = values
                        gkde = gaussian_kde(y)
                        ind = np.linspace(y.min(), y.max(), 1000)
                        ax.plot(ind, gkde.evaluate(ind), **density_kwds)
                    ax.set_xlim(boundaries_list[i])
                else:                # categorical variable
                    values = df[a].value_counts()
                    ax.bar(list(range(df[a].nunique())), values)
            else:
                common = (mask[a] & mask[b]).values
                # two numerical variables
                if a in num_cols and b in num_cols:
                    if i > j:
                        ax.scatter(df[b][common], df[a][common], marker=marker, alpha=alpha, **kwds)
                        # The following 2 lines add the lowess smoothing
                        ys = lowess(df[a][common], df[b][common])
                        ax.plot(ys[:,0], ys[:,1], 'red')
                    else:
                        pearR = df[[a, b]].corr()
                        ax.text(df[b].min(), df[a].min(), 'r = %.4f' % (pearR.iloc[0][1]))
                    ax.set_xlim(boundaries_list[j])
                    ax.set_ylim(boundaries_list[i])
                # two categorical variables
                elif a not in num_cols and b not in num_cols:
                    if i > j:
                        from statsmodels.graphics import mosaicplot
                        mosaicplot.mosaic(df, [b, a], ax, labelizer=lambda k:'')
                # one numerical variable and one categorical variable
                else:
                    if i > j:
                        tol = pd.DataFrame(df[[a, b]])
                        if a in num_cols:
                            label = [ k for k, v in tol.groupby(b) ]
                            values = [ v[a].tolist() for k, v in tol.groupby(b) ]
                            ax.boxplot(values, labels=label)
                        else:
                            label = [ k for k, v in tol.groupby(a) ]
                            values = [ v[b].tolist() for k, v in tol.groupby(a) ]
                            ax.boxplot(values, labels=label, vert=False)

            ax.set_xlabel('')
            ax.set_ylabel('')

            _label_axis(ax, kind='x', label=b, position='bottom', rotate=True)
            _label_axis(ax, kind='y', label=a, position='left')

            if j!= 0:
                ax.yaxis.set_visible(False)
            if i != n-1:
                ax.xaxis.set_visible(False)

    for ax in axes.flat:
        setp(ax.get_xticklabels(), fontsize=8)
        setp(ax.get_yticklabels(), fontsize=8)
    return fig
    

def _label_axis(ax, kind='x', label='', position='top', ticks=True, rotate=False):
    from matplotlib.artist import setp
    if kind == 'x':
        ax.set_xlabel(label, visible=True)
        ax.xaxis.set_visible(True)
        ax.xaxis.set_ticks_position(position)
        ax.xaxis.set_label_position(position)
        if rotate:
            setp(ax.get_xticklabels(), rotation=90)
    elif kind == 'y':
        ax.yaxis.set_visible(True)
        ax.set_ylabel(label, visible=True)
        #ax.set_ylabel(a)
        ax.yaxis.set_ticks_position(position)
        ax.yaxis.set_label_position(position)
    return

def _get_marker_compat(marker):
    import matplotlib.lines as mlines
    import matplotlib as mpl
    if mpl.__version__ < '1.1.0' and marker == '.':
        return 'o'
    if marker not in mlines.lineMarkers:
        return 'o'
    return marker

############################################################################
# PRE-PROCESSING FUNCTIONS
############################################################################
# Missing values
def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"     
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns
   
# Frequency table of every variable:
def freq(df):
    pd.options.display.max_rows = 10
    for col in list(df):
        print(col, flush = True)    
        above_zero = df[col].value_counts()
        print(above_zero[above_zero > 0])
        print('')
        print('---------------------------------------------------------------------------')
        print('')
    return print('')

# Frequency table of each categorical variable:
def freqcat(df):
  pd.options.display.max_rows = 10
  categories = list(df.select_dtypes(['category']))
  booleans = list(df.select_dtypes(['bool']))
  concat = categories + booleans
  for col in concat:
    print(col, flush = True)  
    above_zero = df[col].value_counts()
    print(above_zero[above_zero > 0])
    print('')
    print('---------------------------------------------------------------------------')
    print('')
  return print('')

# Frequency table of every variable in long format:
def lfreq(df):
    pd.options.display.max_rows = 1000
    for col in list(df):
        print(col, flush = True)
        above_zero = df[col].value_counts()
        print(above_zero[above_zero > 0])
        print('')
        print('---------------------------------------------------------------------------')
        print('')
    return print('')

# Frequency table of each categorical variable in long format:
def lfreqcat(df):
  pd.options.display.max_rows = 1000
  categories = list(df.select_dtypes(['category']))
  booleans = list(df.select_dtypes(['bool']))
  objects = list(df.select_dtypes(['object']))
  concat = categories + booleans
  for col in concat:
    print(col, flush = True)
    above_zero = df[col].value_counts()
    print(above_zero[above_zero > 0])
    print('')
    print('---------------------------------------------------------------------------')
    print('')
  return print('')

# Convert to factors:
def to_category(df):
  df[to_cat] = df[to_cat].apply(lambda x: x.astype('category'))
  return df

# Convert to numeric: 
def to_numeric(df):
  df[to_num] = df[to_num].apply(pd.to_numeric, errors='coerce')
  return df

# Convert strings to date: # The date format is: "%Y-%m-%d"
def to_date(df, format_string):
    df[to_dt] = df[to_dt].apply(pd.to_datetime, format=format_string)
    return df
  
# Convert dates in format %b-%Y to numeric: 
def date_to_numeric(df):
  for col in dates:
    print("Converting date variables to years: " + col, flush = True)
    df[col] = df[col].apply(pd.to_datetime, format = '%b-%Y', errors = 'coerce')     
    df[col] = 2018 - df[col].apply(pd.to_datetime, errors = 'coerce').dt.year
  return df

# Convert to string:
def to_string(df):
  df[to_str] = df[to_str].apply(lambda x: x.astype(str))
  return df

def numlevels(df):
    print('CATEGORY')
    print('---------------------------------------------------------------------------')
    print(df.select_dtypes(['category']).apply(lambda x: len(set(x))), flush = True)  
    print('')
    print('BOOLEANS')
    print('---------------------------------------------------------------------------')
    print(df.select_dtypes(['bool']).apply(lambda x: len(set(x))), flush = True)  
    print('')
    print('OBJECT')
    print('---------------------------------------------------------------------------')
    print(df.select_dtypes(['object']).apply(lambda x: len(set(x))), flush = True)  
    return print('')

# Impute Missing Values:
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value 
        in column.
        Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
# Missing Values Imputation
def scatter_matrix_all(frame, alpha=0.5, figsize=None, grid=False, diagonal='hist', marker='.', density_kwds=None, hist_kwds=None, range_padding=0.05, **kwds):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.artist import setp
    import pandas.core.common as com
    from pandas.compat import range, lrange, lmap, map, zip
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    df = frame
    num_cols = frame._get_numeric_data().columns.values
    n = df.columns.size
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=figsize, squeeze=False)

    # no gaps between subplots
    fig.subplots_adjust(wspace=0, hspace=0)

    mask = com.notnull(df)
    marker = _get_marker_compat(marker)

    hist_kwds = hist_kwds or {}
    density_kwds = density_kwds or {}

    # workaround because `c='b'` is hardcoded in matplotlibs scatter method
    kwds.setdefault('c', plt.rcParams['patch.facecolor'])

    boundaries_list = []
    for a in df.columns:
        if a in num_cols:
            values = df[a].values[mask[a].values]
        else:
            values = df[a].value_counts()
        rmin_, rmax_ = np.min(values), np.max(values)
        rdelta_ext = (rmax_ - rmin_) * range_padding / 2.
        boundaries_list.append((rmin_ - rdelta_ext, rmax_+ rdelta_ext))

    for i, a in zip(lrange(n), df.columns):
        for j, b in zip(lrange(n), df.columns):
            ax = axes[i, j]

            if i == j:
                if a in num_cols:    # numerical variable
                    values = df[a].values[mask[a].values]
                    # Deal with the diagonal by drawing a histogram there.
                    if diagonal == 'hist':
                        ax.hist(values, **hist_kwds)
                    elif diagonal in ('kde', 'density'):
                        from scipy.stats import gaussian_kde
                        y = values
                        gkde = gaussian_kde(y)
                        ind = np.linspace(y.min(), y.max(), 1000)
                        ax.plot(ind, gkde.evaluate(ind), **density_kwds)
                    ax.set_xlim(boundaries_list[i])
                else:                # categorical variable
                    values = df[a].value_counts()
                    ax.bar(list(range(df[a].nunique())), values)
            else:
                common = (mask[a] & mask[b]).values
                # two numerical variables
                if a in num_cols and b in num_cols:
                    if i > j:
                        ax.scatter(df[b][common], df[a][common], marker=marker, alpha=alpha, **kwds)
                        # The following 2 lines add the lowess smoothing
                        ys = lowess(df[a][common], df[b][common])
                        ax.plot(ys[:,0], ys[:,1], 'red')
                    else:
                        pearR = df[[a, b]].corr()
                        ax.text(df[b].min(), df[a].min(), 'r = %.4f' % (pearR.iloc[0][1]))
                    ax.set_xlim(boundaries_list[j])
                    ax.set_ylim(boundaries_list[i])
                # two categorical variables
                elif a not in num_cols and b not in num_cols:
                    if i > j:
                        from statsmodels.graphics import mosaicplot
                        mosaicplot.mosaic(df, [b, a], ax, labelizer=lambda k:'')
                # one numerical variable and one categorical variable
                else:
                    if i > j:
                        tol = pd.DataFrame(df[[a, b]])
                        if a in num_cols:
                            label = [ k for k, v in tol.groupby(b) ]
                            values = [ v[a].tolist() for k, v in tol.groupby(b) ]
                            ax.boxplot(values, labels=label)
                        else:
                            label = [ k for k, v in tol.groupby(a) ]
                            values = [ v[b].tolist() for k, v in tol.groupby(a) ]
                            ax.boxplot(values, labels=label, vert=False)

            ax.set_xlabel('')
            ax.set_ylabel('')

            _label_axis(ax, kind='x', label=b, position='bottom', rotate=True)
            _label_axis(ax, kind='y', label=a, position='left')

            if j!= 0:
                ax.yaxis.set_visible(False)
            if i != n-1:
                ax.xaxis.set_visible(False)

    for ax in axes.flat:
        setp(ax.get_xticklabels(), fontsize=8)
        setp(ax.get_yticklabels(), fontsize=8)
    return fig
    

def _label_axis(ax, kind='x', label='', position='top', ticks=True, rotate=False):
    from matplotlib.artist import setp
    if kind == 'x':
        ax.set_xlabel(label, visible=True)
        ax.xaxis.set_visible(True)
        ax.xaxis.set_ticks_position(position)
        ax.xaxis.set_label_position(position)
        if rotate:
            setp(ax.get_xticklabels(), rotation=90)
    elif kind == 'y':
        ax.yaxis.set_visible(True)
        ax.set_ylabel(label, visible=True)
        #ax.set_ylabel(a)
        ax.yaxis.set_ticks_position(position)
        ax.yaxis.set_label_position(position)
    return

def _get_marker_compat(marker):
    import matplotlib.lines as mlines
    import matplotlib as mpl
    if mpl.__version__ < '1.1.0' and marker == '.':
        return 'o'
    if marker not in mlines.lineMarkers:
        return 'o'
    return marker

# PCA Dimensionality Reduction
def pca_summary(pca, standardised_data, out=True):
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(list(zip(a, b, c)), index=names, columns=columns)
    if out:
        print("Importance of components:")
        display(summary)
    return summary

# Accumulative importance and number of components graph
def screeplot(pca, standardised_values):
    y = np.std(pca.transform(standardised_values), axis=0)**2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    plt.show()

# Graph any PCA of the components desired: "first" parameter indicates Component in x-axis
# "second" parameter indicates Component in y-axis 
def pca_scatter(pca, standardised_values, classifs, first, second):
    foo = pca.transform(standardised_values)
    bar = pd.DataFrame(list(zip(foo[:, first-1], foo[:, second-1], classifs)), columns=[str("PC"+str(first)), str("PC"+str(second)), "Class"])
    sns.lmplot(str("PC"+str(first)), str("PC"+str(second)), bar, hue="Class", fit_reg=False)

# PARETO CHART IN PYTHON
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
df = pd.DataFrame({'country': [177.0, 7.0, 4.0, 2.0, 2.0, 1.0, 1.0, 1.0]})
df.index = ['USA', 'Canada', 'Russia', 'UK', 'Belgium', 'Mexico', 'Germany', 'Denmark']
df = df.sort_values(by='country',ascending=False)
df["cumpercentage"] = df["country"].cumsum()/df["country"].sum()*100
print(df)
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1,1,1)
ax.bar(df.index, df["country"], color="C0")
ax2 = ax.twinx()
ax2.plot(df.index, df["cumpercentage"], color="C1", marker="D", ms=7)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")
plt.show()

# ADDING VERTICAL LINES TO PLOTS
# ys is the value where we want the line to be marked.
def axhlines(ys, ax=None, **plot_kwargs):
    """
    Draw horizontal lines across plot
    :param ys: A scalar, list, or 1D array of vertical offsets
    :param ax: The axis (or none to use gca)
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    if ax is None:
        ax = plt.gca()
    ys = np.array((ys, ) if np.isscalar(ys) else ys, copy=False)
    lims = ax.get_xlim()
    y_points = np.repeat(ys[:, None], repeats=3, axis=1).flatten()
    x_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(ys), axis=0).flatten()
    plot = ax.plot(x_points, y_points, scalex = False, **plot_kwargs)
    return plot

# ADDING HORIZONTAL LINES TO PLOTS
# xs is the value where we want the line to be marked.
def axvlines(xs, ax=None, **plot_kwargs):
    """
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param ax: The axis (or none to use gca)
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    if ax is None:
        ax = plt.gca()
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    lims = ax.get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(xs), axis=0).flatten()
    plot = ax.plot(x_points, y_points, scaley = False, **plot_kwargs)
    return plot
def scatter_matrix_all(frame, alpha=0.5, figsize=None, grid=False, diagonal='hist', marker='.', density_kwds=None, hist_kwds=None, range_padding=0.05, **kwds):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.artist import setp
    import pandas.core.common as com
    from pandas.compat import range, lrange, lmap, map, zip
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    df = frame
    num_cols = frame._get_numeric_data().columns.values
    n = df.columns.size
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=figsize, squeeze=False)

    # no gaps between subplots
    fig.subplots_adjust(wspace=0, hspace=0)

    mask = com.notnull(df)
    marker = _get_marker_compat(marker)

    hist_kwds = hist_kwds or {}
    density_kwds = density_kwds or {}

    # workaround because `c='b'` is hardcoded in matplotlibs scatter method
    kwds.setdefault('c', plt.rcParams['patch.facecolor'])

    boundaries_list = []
    for a in df.columns:
        if a in num_cols:
            values = df[a].values[mask[a].values]
        else:
            values = df[a].value_counts()
        rmin_, rmax_ = np.min(values), np.max(values)
        rdelta_ext = (rmax_ - rmin_) * range_padding / 2.
        boundaries_list.append((rmin_ - rdelta_ext, rmax_+ rdelta_ext))

    for i, a in zip(lrange(n), df.columns):
        for j, b in zip(lrange(n), df.columns):
            ax = axes[i, j]

            if i == j:
                if a in num_cols:    # numerical variable
                    values = df[a].values[mask[a].values]
                    # Deal with the diagonal by drawing a histogram there.
                    if diagonal == 'hist':
                        ax.hist(values, **hist_kwds)
                    elif diagonal in ('kde', 'density'):
                        from scipy.stats import gaussian_kde
                        y = values
                        gkde = gaussian_kde(y)
                        ind = np.linspace(y.min(), y.max(), 1000)
                        ax.plot(ind, gkde.evaluate(ind), **density_kwds)
                    ax.set_xlim(boundaries_list[i])
                else:                # categorical variable
                    values = df[a].value_counts()
                    ax.bar(list(range(df[a].nunique())), values)
            else:
                common = (mask[a] & mask[b]).values
                # two numerical variables
                if a in num_cols and b in num_cols:
                    if i > j:
                        ax.scatter(df[b][common], df[a][common], marker=marker, alpha=alpha, **kwds)
                        # The following 2 lines add the lowess smoothing
                        ys = lowess(df[a][common], df[b][common])
                        ax.plot(ys[:,0], ys[:,1], 'red')
                    else:
                        pearR = df[[a, b]].corr()
                        ax.text(df[b].min(), df[a].min(), 'r = %.4f' % (pearR.iloc[0][1]))
                    ax.set_xlim(boundaries_list[j])
                    ax.set_ylim(boundaries_list[i])
                # two categorical variables
                elif a not in num_cols and b not in num_cols:
                    if i > j:
                        from statsmodels.graphics import mosaicplot
                        mosaicplot.mosaic(df, [b, a], ax, labelizer=lambda k:'')
                # one numerical variable and one categorical variable
                else:
                    if i > j:
                        tol = pd.DataFrame(df[[a, b]])
                        if a in num_cols:
                            label = [ k for k, v in tol.groupby(b) ]
                            values = [ v[a].tolist() for k, v in tol.groupby(b) ]
                            ax.boxplot(values, labels=label)
                        else:
                            label = [ k for k, v in tol.groupby(a) ]
                            values = [ v[b].tolist() for k, v in tol.groupby(a) ]
                            ax.boxplot(values, labels=label, vert=False)

            ax.set_xlabel('')
            ax.set_ylabel('')

            _label_axis(ax, kind='x', label=b, position='bottom', rotate=True)
            _label_axis(ax, kind='y', label=a, position='left')

            if j!= 0:
                ax.yaxis.set_visible(False)
            if i != n-1:
                ax.xaxis.set_visible(False)

    for ax in axes.flat:
        setp(ax.get_xticklabels(), fontsize=8)
        setp(ax.get_yticklabels(), fontsize=8)
    return fig
    

def _label_axis(ax, kind='x', label='', position='top', ticks=True, rotate=False):
    from matplotlib.artist import setp
    if kind == 'x':
        ax.set_xlabel(label, visible=True)
        ax.xaxis.set_visible(True)
        ax.xaxis.set_ticks_position(position)
        ax.xaxis.set_label_position(position)
        if rotate:
            setp(ax.get_xticklabels(), rotation=90)
    elif kind == 'y':
        ax.yaxis.set_visible(True)
        ax.set_ylabel(label, visible=True)
        #ax.set_ylabel(a)
        ax.yaxis.set_ticks_position(position)
        ax.yaxis.set_label_position(position)
    return

def _get_marker_compat(marker):
    import matplotlib.lines as mlines
    import matplotlib as mpl
    if mpl.__version__ < '1.1.0' and marker == '.':
        return 'o'
    if marker not in mlines.lineMarkers:
        return 'o'
    return marker

############################################################################
# PRE-PROCESSING FUNCTIONS
############################################################################
# Frequency table of every variable:
def freq(df):
    pd.options.display.max_rows = 10
    for col in list(df):
        print(col, flush = True)    
        above_zero = df[col].value_counts()
        print(above_zero[above_zero > 0])
        print('')
        print('---------------------------------------------------------------------------')
        print('')
    return print('')

# Frequency table of each categorical variable:
def freqcat(df):
  pd.options.display.max_rows = 10
  categories = list(df.select_dtypes(['category']))
  booleans = list(df.select_dtypes(['bool']))
  objects = list(df.select_dtypes(['object']))
  concat = categories + booleans + objects
  for col in concat:
    print(col, flush = True)  
    above_zero = df[col].value_counts()
    print(above_zero[above_zero > 0])
    print('')
    print('---------------------------------------------------------------------------')
    print('')
  return print('')

# Frequency table of every variable in long format:
def lfreq(df):
    pd.options.display.max_rows = 1000
    for col in list(df):
        print(col, flush = True)
        above_zero = df[col].value_counts()
        print(above_zero[above_zero > 0])
        print('')
        print('---------------------------------------------------------------------------')
        print('')
    return print('')

# Frequency table of each categorical variable in long format:
def lfreqcat(df):
  pd.options.display.max_rows = 1000
  categories = list(df.select_dtypes(['category']))
  booleans = list(df.select_dtypes(['bool']))
  objects = list(df.select_dtypes(['object']))
  concat = categories + booleans + objects
  for col in concat:
    print(col, flush = True)
    above_zero = df[col].value_counts()
    print(above_zero[above_zero > 0])
    print('')
    print('---------------------------------------------------------------------------')
    print('')
  return print('')

# Convert to factors:
def to_string(df, to_str):
    df[to_str] = df[to_str].apply(lambda x: x.astype('category'))
    return df

# Convert to numeric: 
def to_numeric(df, to_num):
    df[to_num] = df[to_num].apply(pd.to_numeric, errors='coerce')
    return df

# Convert dates in format %b-%Y to numeric: 
def date_to_numeric(df):
  for col in dates:
    print("Converting date variables to years: " + col, flush = True)
    df[col] = df[col].apply(pd.to_datetime, format = '%b-%Y', errors = 'coerce')     
    df[col] = 2018 - df[col].apply(pd.to_datetime, errors = 'coerce').dt.year
  return df

# Convert to string:
def to_string(df):
  df[to_str] = df[to_str].apply(lambda x: x.astype(str))
  return df

def numlevels(df):
    print('CATEGORY')
    print('---------------------------------------------------------------------------')
    print(df.select_dtypes(['category']).apply(lambda x: len(set(x))), flush = True)  
    print('')
    print('BOOLEANS')
    print('---------------------------------------------------------------------------')
    print(df.select_dtypes(['bool']).apply(lambda x: len(set(x))), flush = True)  
    print('')
    print('OBJECT')
    print('---------------------------------------------------------------------------')
    print(df.select_dtypes(['object']).apply(lambda x: len(set(x))), flush = True)  
    return print('')

# Impute Missing Values:
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value 
        in column.
        Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
# Missing Values Imputation
def scatter_matrix_all(frame, alpha=0.5, figsize=None, grid=False, diagonal='hist', marker='.', density_kwds=None, hist_kwds=None, range_padding=0.05, **kwds):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.artist import setp
    import pandas.core.common as com
    from pandas.compat import range, lrange, lmap, map, zip
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    df = frame
    num_cols = frame._get_numeric_data().columns.values
    n = df.columns.size
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=figsize, squeeze=False)

    # no gaps between subplots
    fig.subplots_adjust(wspace=0, hspace=0)

    mask = com.notnull(df)
    marker = _get_marker_compat(marker)

    hist_kwds = hist_kwds or {}
    density_kwds = density_kwds or {}

    # workaround because `c='b'` is hardcoded in matplotlibs scatter method
    kwds.setdefault('c', plt.rcParams['patch.facecolor'])

    boundaries_list = []
    for a in df.columns:
        if a in num_cols:
            values = df[a].values[mask[a].values]
        else:
            values = df[a].value_counts()
        rmin_, rmax_ = np.min(values), np.max(values)
        rdelta_ext = (rmax_ - rmin_) * range_padding / 2.
        boundaries_list.append((rmin_ - rdelta_ext, rmax_+ rdelta_ext))

    for i, a in zip(lrange(n), df.columns):
        for j, b in zip(lrange(n), df.columns):
            ax = axes[i, j]

            if i == j:
                if a in num_cols:    # numerical variable
                    values = df[a].values[mask[a].values]
                    # Deal with the diagonal by drawing a histogram there.
                    if diagonal == 'hist':
                        ax.hist(values, **hist_kwds)
                    elif diagonal in ('kde', 'density'):
                        from scipy.stats import gaussian_kde
                        y = values
                        gkde = gaussian_kde(y)
                        ind = np.linspace(y.min(), y.max(), 1000)
                        ax.plot(ind, gkde.evaluate(ind), **density_kwds)
                    ax.set_xlim(boundaries_list[i])
                else:                # categorical variable
                    values = df[a].value_counts()
                    ax.bar(list(range(df[a].nunique())), values)
            else:
                common = (mask[a] & mask[b]).values
                # two numerical variables
                if a in num_cols and b in num_cols:
                    if i > j:
                        ax.scatter(df[b][common], df[a][common], marker=marker, alpha=alpha, **kwds)
                        # The following 2 lines add the lowess smoothing
                        ys = lowess(df[a][common], df[b][common])
                        ax.plot(ys[:,0], ys[:,1], 'red')
                    else:
                        pearR = df[[a, b]].corr()
                        ax.text(df[b].min(), df[a].min(), 'r = %.4f' % (pearR.iloc[0][1]))
                    ax.set_xlim(boundaries_list[j])
                    ax.set_ylim(boundaries_list[i])
                # two categorical variables
                elif a not in num_cols and b not in num_cols:
                    if i > j:
                        from statsmodels.graphics import mosaicplot
                        mosaicplot.mosaic(df, [b, a], ax, labelizer=lambda k:'')
                # one numerical variable and one categorical variable
                else:
                    if i > j:
                        tol = pd.DataFrame(df[[a, b]])
                        if a in num_cols:
                            label = [ k for k, v in tol.groupby(b) ]
                            values = [ v[a].tolist() for k, v in tol.groupby(b) ]
                            ax.boxplot(values, labels=label)
                        else:
                            label = [ k for k, v in tol.groupby(a) ]
                            values = [ v[b].tolist() for k, v in tol.groupby(a) ]
                            ax.boxplot(values, labels=label, vert=False)

            ax.set_xlabel('')
            ax.set_ylabel('')

            _label_axis(ax, kind='x', label=b, position='bottom', rotate=True)
            _label_axis(ax, kind='y', label=a, position='left')

            if j!= 0:
                ax.yaxis.set_visible(False)
            if i != n-1:
                ax.xaxis.set_visible(False)

    for ax in axes.flat:
        setp(ax.get_xticklabels(), fontsize=8)
        setp(ax.get_yticklabels(), fontsize=8)
    return fig
    

def _label_axis(ax, kind='x', label='', position='top', ticks=True, rotate=False):
    from matplotlib.artist import setp
    if kind == 'x':
        ax.set_xlabel(label, visible=True)
        ax.xaxis.set_visible(True)
        ax.xaxis.set_ticks_position(position)
        ax.xaxis.set_label_position(position)
        if rotate:
            setp(ax.get_xticklabels(), rotation=90)
    elif kind == 'y':
        ax.yaxis.set_visible(True)
        ax.set_ylabel(label, visible=True)
        #ax.set_ylabel(a)
        ax.yaxis.set_ticks_position(position)
        ax.yaxis.set_label_position(position)
    return

def _get_marker_compat(marker):
    import matplotlib.lines as mlines
    import matplotlib as mpl
    if mpl.__version__ < '1.1.0' and marker == '.':
        return 'o'
    if marker not in mlines.lineMarkers:
        return 'o'
    return marker

# PCA Dimensionality Reduction
def pca_summary(pca, standardised_data, out=True):
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(list(zip(a, b, c)), index=names, columns=columns)
    if out:
        print("Importance of components:")
        display(summary)
    return summary

# Accumulative importance and number of components graph
def screeplot(pca, standardised_values):
    y = np.std(pca.transform(standardised_values), axis=0)**2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    plt.show()

# Graph any PCA of the components desired: "first" parameter indicates Component in x-axis
# "second" parameter indicates Component in y-axis 
def pca_scatter(pca, standardised_values, classifs, first, second):
    foo = pca.transform(standardised_values)
    bar = pd.DataFrame(list(zip(foo[:, first-1], foo[:, second-1], classifs)), columns=[str("PC"+str(first)), str("PC"+str(second)), "Class"])
    sns.lmplot(str("PC"+str(first)), str("PC"+str(second)), bar, hue="Class", fit_reg=False)

# VIF
pd.Series([variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])], index = Xc.columns)

# SORT DATA FRAME WITH COLUMNS IN ALPHABETICAL ORDER
def sortdf(df):
    df.sort_index(axis = 1, inplace = True)
    return(df)

# PRINT ALL ROWS OF DATA FRAMES 
def maxrows():
    pd.options.display.max_rows = None
    return()

# OBTAIN ALL TYPES OF COLUMN NAMES IN ALPHABETICAL ORDER
def types(df):
    return(print(df.dtypes.sort_index()))

def variables(df):
    return((sorted(list(df))))

def nmissings(df):
    return(sum(df.isnull().sum()))

# AUTOS SAMPLE SCRIPT
############################################################################
# IMPORT OTHER SCRIPTS
############################################################################
exec(open("C:/ANIBAL/FORMATION/PERSONAL/2_autos_uciml/scripts/1_libraries_autos.py").read())
exec(open("C:/ANIBAL/FORMATION/PERSONAL/2_autos_uciml/scripts/2_properties_autos.py").read())
exec(open("C:/ANIBAL/FORMATION/PERSONAL/2_autos_uciml/scripts/3_functions_autos.py").read())
exec(open("C:/ANIBAL/FORMATION/PERSONAL/2_autos_uciml/scripts/4_environ_autos.py").read())

############################################################################
# IMPORT DATA
############################################################################
# Import data table:
df = pd.read_csv(file_dir + file_name, sep = ',', header = None, encoding = 'latin-1', low_memory = False)
# Import column_names table:
df_column_names = pd.read_csv(file_dir + file_name_cols, sep = ',', header = None, encoding = 'latin-1', low_memory = False)
# Transpose column_names from 1st row to 1st column:
df_column_names = df_column_names.transpose() 
list_names = df_column_names[0].tolist()
# Assign header:
df.columns = list_names
# We substitute hyphens for underscores in column names:
df.columns = df.columns.str.replace('-', '_')

############################################################################
# FAST DATA QUALITY CHECK
############################################################################
df.isnull().sum()/len(df)*100 # The data set is completely clean

############################################################################
# FAST PRE-PROCESSING
############################################################################
# 1. Visual inspection
print(df) # Or doing it with the GUI

# 2. Drop duplicated rows:
df = df.drop_duplicates(keep = 'first') # OK
# 3. Drop rows with 70% missing values or more:
df = df.dropna(thresh = round(len(df.columns) * (1-thresh_rows) , 0))

# 4. Drop duplicated columns (equal to other columns in all its registry values):
df = df.loc[:,~df.columns.duplicated()] # Only works for dataframes whose columns have equal names
df = df.T.drop_duplicates().T # Memory SUPER intensive - It refers to the values
# 5. Drop columns with 70% of missing values or more:
df = df[df.columns[df.isnull().mean() < thresh_cols]] # OK
# 6. Drop columns without variance:
df = df.loc[:,df.apply(pd.Series.nunique) != 1] # Seems that this option is better coz´it also supports 'category' type

# 7. Organize dataframe columns alphabetically:
df = df.reindex(sorted(df.columns), axis=1)

############################################################################
# SECOND DATA QUALITY CHECK
############################################################################
df.isnull().sum()/len(df)*100 # The data set is completely clean

############################################################################
# ANALYZING FACTOR VARIABLES I: Changing Type of Variables
############################################################################
# 1. Correct variable type setting:
df.dtypes
freq(df)

# Variables

# To String:
# aspiration -> string
# body-style -> string
# drive-wheels -> string
# engine-location -> string
# engine-type -> string
# fuel-system -> string
# fuel-type -> string
# make -> string
# num-of-cylinders -> string
# num-of-doors -> string

# To Float64:
# bore -> float64
# city-mpg -> float64
# compression-ratio -> float64
# curb-weight -> float64
# engine-size -> float64
# height -> float64
# highway-mpg -> float64
# horsepower -> float64
# length -> float64
# normalized-losses -> float64
# peak-rpm -> float64
# price -> float64
# stroke -> float64
# symboling -> float64
# wheel-base -> float64
# width -> float64

# 1.2) To Numeric type:
df = to_numeric(df)
df.dtypes

############################################################################
# ANALYZING FACTOR VARIABLES II: Checking value-attribute correspondence
############################################################################
# Show all categorical attributes and its values:
lfreqcat(df)
# Recoding of variables
# Replace missing values for "?":
df.replace(to_replace = '?', value = np.nan, inplace = True)
# Recoding of the response variable:
df.loc[df.symboling > 0 , 'symboling'] = 1
df.loc[df.symboling <= 0 , 'symboling'] = 0
# Checking cardinality of categorical variables:
numlevels(df) # OK
# Checking correspondence of categorical variables and its values
lfreqcat(df)

############################################################################
# ANALYZING FACTOR VARIABLES III: Eliminating/editing high cardinality factor variables (number of levels)
############################################################################
# List number of levels in each categorical variable:
df.select_dtypes(['category']).apply(lambda x: len(set(x)))
numlevels(df)
# Everything is OK

# YAML FILES IMPORT
import yaml # Requires PyYAML installation
# DEFINE ROOT FOLDER:
root_folder = Path(__file__).parents[2] #  Requires: from pathlib import Path
# IMPORT YAML FILE:
yaml_rel_route = "src/properties/properties.yaml"
with open(os.path.join(Path(root_folder), Path(yaml_rel_route))) as f:
    properties = yaml.load(f, Loader=yaml.BaseLoader)
filename = os.path.join(root_folder, properties['directories_routes']['output_models'],
                        properties['model_instances']['rf_naive_win_lost_min_preproc_c_a_l_all_train_objects'])
properties[directories_routes']['complete_path_to_model'] # Example of accessing a YAML item

###################################################### JUPYTER NOTEBOOK ################################################################
# LOAD EXTERNAL JUPYTER NOTEBOOK IN A CELL / RUN JUPYTER NOTEBOOK INSIDE A NOTEBOOK
%run name_of_notebook.ipynb
%run ./name_of_notebook.ipynb # Other way of doing the same

# CENTERING TEXT IN MARKDOWN / CENTER TEXT 
<p style="text-align: center;">Centered text</p>

# ADD PHOTOS / IMAGES / JPG / JPEG / IN MARKDOWN OR JUPYTER NOTEBOOK
from IPython.display import Image
Image(filename="C:/ANIBAL/FORMATION/PERSONAL/feature_engineering_for_machine_learning/name_of_file.PNG")

# PROPERTIES CORRECTED FUNCTION
def props_import(yaml_location_path):
    """
    Function that imports properties objects of the project.

    Returns
    -------
    dict: a Python dictionary with 6 keys: "repo", "yaml_location", "root", "yaml_file", "properties_general", "props".
    """
    # We use git library to get the source directory of repo (A part):
    repo = git.Repo(search_parent_directories=True)
    # We transform "repo" object which is in git format to Windows path string format:
    root = Path(repo.working_tree_dir)
    # We create the extension where the "properties.yaml" file exists (B part)
    yaml_location = yaml_location_path
    # Connecting A and B parts:
    yaml_file = os.path.join(root, yaml_location)
    # Import properties.yaml file:
    with open(yaml_file) as file:
        properties_general = yaml.load(file, Loader=yaml.BaseLoader)
    # Create all properties dictionary:
    props = dict.fromkeys(properties_general["properties"])
    # Load other properties:
    for key in properties_general["properties"].keys():
        with open(os.path.join(Path(root), Path(properties_general["properties"][str(key)]))) as file:
            props[str(key)] = yaml.load(file, Loader=yaml.BaseLoader)
    return {"repo": repo,
            "yaml_location": yaml_location,
            "root": root,
            "yaml_file": yaml_file,
            "properties_general": properties_general,
            "props": props}
