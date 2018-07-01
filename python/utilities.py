# INSTALL PACKAGES IN ANACONDA:
# Standard way:
conda install name_of_package
# If not available from current channel:
conda install -c conda-forge name_of_package

# CREATION OF A MOCK DATAFRAME
# Option 1: 
x = [1,1000,1001]
y = [200,300,400]
cat = ['first','second','third']
df = pd.DataFrame(dict(speed = x, price = y, place = cat))

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

# READ DATA / IMPORT DATA / READ CSV FILES / IMPORT CSV
# Import CSV:
df = pd.read_csv(file_dir + file_name, sep = ',', header = 0, encoding = 'latin-1', low_memory = False)
# use dtypes argument when all rows are consistent in type: i.e. dtype={'user_id': int}
# Import parquet files:
df = pd.read_parquet(file_dir + file_name_perf, engine='pyarrow')

# MERGE / JOIN DATA FRAMES
df = pd.merge(df1, df2, on = ['variable_1', 'variable_2'], how = 'inner')
# how can be changed to 'outer', 'left' or right

# DATA TYPE CONVERSION
# Convert all columns to numeric type:
df = df.apply(pd.to_numeric, errors='coerce')

# MISSINGS / DATA QUALITY ASSESSMENT
# Count number of missings for a column:
df['hardship_amount'].isnull().sum()
# Show number of missings for each column:
df.isnull().sum()
# Show number of missings per column in percentage:
df.isnull().sum()/len(df)*100

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

# SELECT / FILTER DATA / FILTER A DATASET / FILTER DATAFRAME / EXCLUDE / NOT SELECT
# Index numeric, column by name:
df.loc[df.index[number_desired_of_index], 'name_of_column']
# Select all dataframe except for one column
df.loc[:, df.columns != 'name_of_column_to_exclude']
# Obtain all values not included in a list of values: list_with_values_of_variable_to_filter
dfx = df[~df['name_of_variable'].isin(list_with_values_of_variable_to_filter)] 

# CREATE NEW COLUMN / ADD NEW COLUMN / ADD NEW FEATURE / ADD NEW VARIABLE
df['name_of_new_column'] = pd.Series(np.nan , index = df.index)

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

# COLUMN NAMES REPLACE STRINGS / 
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

# DATE / DATETIME
# Convert object to datetime/date type:
df['name_of_column'] =  pd.to_datetime(df['name_of_column'], format='%d%b%Y:%H:%M:%S.%f')


























Frame(dict(x=x, y=y))
df_completeness_plms.columns = ['variable', 'complete_metric']
# Create vertical bar plot: factorplot:
g = sns.factorplot("variable","complete_metric", data=df_completeness_plms,kind="bar",palette="Blues",size=6,aspect=2,legend_out=False)
g.set_xticklabels(rotation=90)

# Create horizontal bar plot: barplot
f, ax = plt.subplots(figsize=(18, 25)) # Set size of the graph
sns.set(font_scale = 2) # Set size of the variable labels in each axis, as well as the axis values
sns.barplot(x = "complete_metric", y = "variable", data = df_completeness_dqss, label="Total", palette=sns.color_palette("husl", 20)) # Create the barplot
plt.xlabel("Completeness") # Assign name of the x label
plt.ylabel("Variables") # Assign name of the y label
plt.title("Good results in 'Completeness' metric") # Assign title of the graph
plt.show(sns.barplot) # Show everything

# blue horizontal bar plot
f, ax = plt.subplots(figsize=(18, 25)) # Set size of the graph
sns.set(font_scale = 2) # Set size of the variable labels in each axis, as well as the axis values
sns.barplot(x = "complete_metric", y = "variable", data = df_completeness_dqss, label="Total", color='b') # Create the barplot
plt.xlabel("Completeness") # Assign name of the x label
plt.ylabel("Variables") # Assign name of the y label
plt.title("Good results in DQSS 'Completeness' metric") # Assign title of the graph

# End a process in terminal
# Click in terminal and write: "Ctrl + C"
# See which is our current directory 
os.getcwd()
# Change directory:
os.chdir(route_of_new_directory)
# Import data:
df = pd.read_csv(file_dir + file_name, sep = ',', header = 0, encoding = 'latin-1', low_memory = False)
# use dtypes argument when all rows are consistent in type: i.e. dtype={'user_id': int}
# See names of columns
list(df)
# Assign names to rows index
df.index = df['id']
# Select categorical variables / select variables of type category / Names of columns with type Category:
list(df.select_dtypes(['category']))

# number of levels of categorical variables
df.select_dtypes(['category']).apply(lambda x: len(set(x)))
df.select_dtypes(['object']).apply(lambda x: len(set(x)))
df.select_dtypes(['bool']).apply(lambda x: len(set(x)))


# change data type, convert type to string
df['id']= df['id'].astype(str)
# change data type, convert type to string
df['id']= df['id'].astype('str')

# Change types using a loop
for col in ['parks', 'playgrounds', 'sports', 'roading']:
    public[col] = public[col].astype('category')
    
# Frequency table for any variable    
df['hardship_amount'].value_counts(dropna=False) #It drops Na autmatically

# check what or which registries are equal to a value
df.loc[df['hardship_amount'] == '536.81']

# Block cooment es para comentar todo lo que escojes
# Ctrl + 4 in spyder

# frequency tables
df[col].value_counts()

# Convert object variables to category:
df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))

# Assign names to rows index:
df.index = df['id']
# Check:
list(df.index)

# Select all variables that arent several types of columns:
list(df.select_dtypes(exclude=["number","bool_","object_"]))

# select object types of variables:
list(df.loc[:, df.dtypes == object])

# select categorical variables / object types of variables:
list(df.loc[:, df.dtypes == object])

df_plms.packages.quantile(0.0) # 10th percentile
df_plms.packages.quantile(0.1) # 10th percentile
df_plms.packages.quantile(0.2) # 10th percentile
df_plms.packages.quantile(0.3) # 10th percentile
df_plms.packages.quantile(0.4) # 10th percentile
df_plms.packages.quantile(0.5) # 10th percentile
df_plms.packages.quantile(0.6) # 10th percentile
df_plms.packages.quantile(0.7) # 10th percentile
df_plms.packages.quantile(0.8) # 10th percentile
df_plms.packages.quantile(0.9) # 10th percentile
df_plms.packages.quantile(1.0) # 10th percentile

# Assign names to rows index:
df.index = df['id']

# Watch every columns data type:
df.dtypes
# Convert all columns to string:
df = df.astype(str)
#Obtain column names / names of columns
list(df.columns.values)
list(df)
# Watch row names
list(df.index)

# Assign names to rows
df.index['Row 2':'Row 5'] 
# See names of columns
list(df)
# Assign names to rows index
df.index = df['id']
 
# Example of FOR
for j in range(12, 25):
	df.iloc[i][j] = df.iloc[i][j+1]
# Interrupt Python console:
Ctrl + C
    
# Introduce an na in python:
np.nan

# % of missings in each column. The values are in X/100 not X/1 
df.isnull().sum()/len(df)*100

# nan is the same as NaN and as NAN

# "palette" argument can be interchanged by "color" argument in library seaborn
# "palette" argument in barplot (seaborn library) valid values:
'Blues_d'
'muted'  
'RdBu'
'Set1'
n = number_of_different_colors_you_want_in_your_barplot
sns.color_palette("husl", n)
# Ex: barplot with 20 bars:
n = 20
sns.color_palette("husl", n)

# Access to the cell of a data frame
list(df.index)
# See names of columns
list(df)






# creating factor plots
x = [1,1000,1001]
y = [200,300,400]
cat = ['first','second','third']
df = pd.DataFrame(dict(x=x, y=y,cat=cat))
sns.factorplot("x","y", data=df,kind="bar",palette="Blues",size=6,aspect=2,legend_out=False)

#Create a dataframe in python
    d = {
    'age': [1, 'a', 3, 4, 5, 6, 7, np.nan, np.nan, 10, 1, 1, np.nan], 
     'name': [1, 'a', 3, 4, 5, 6, 6, np.nan, np.nan, 10, 1, 1, np.nan],
     'height': [1, 'a', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10, 1, 1, np.nan],
     'education': [4, 'a', 4, 4, 4, 4, 4,4,4,  4, 1, 1, np.nan],
     'nationality': [4, 'a', 4, 4, 4, 4, 4,np.nan,np.nan,  4, 1, 1, np.nan], 
     'sex': [1, 'a', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10, 1, 1, np.nan],
     'preference': ['a','a','a', 'a', 'a', 'a', 'a', np.nan, np.nan, 'a', 1, 1, np.nan],
     'income': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 1, np.nan],
     'col9': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 1, 1, np.nan],
     'col10': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, np.nan],
     'col11': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, np.nan],
     'col12': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, np.nan],
     'col13': [1,1,1,1,1,1,1,1,1,1, 1, 1, np.nan],
     'col14': ['a','a','a','a','a','a','a','a','a','a', 1, 1,1]
     }
df = pd.DataFrame(data=d)
print(df)

# change object to category
# Change data type to 'category':
plms[plms.select_dtypes(['object']).columns] = plms.select_dtypes(['object']).apply(lambda x: x.astype('category'))
plms.dtypes


#Create a dataframe in python
    d = {
    'age': [32, 41, 35, 43, 57, 63, 73, np.nan, np.nan, 20, 45, 19, np.nan], 
     'name': [1, 'a', 3, 4, 5, 6, 6, np.nan, np.nan, 10, 1, 1, np.nan],
     'height': [1, 'a', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10, 1, 1, np.nan],
     'education': ['law', 'engineering', 'arts', 'economics', 'medicine', 'architecture', 'literature', 'photography', 'cuisine', 'law', 'business_mgmt', 'sport', 'psychology'],
     'nationality': ['india', 'colombia', 'usa', 'sweden', 'italy', 'denmark', 'canada', 'luxembourg', 'england', np.nan, 'south korea', 'australia', np.nan], 
     'sex': ['m', 'm', 'f', 'f', 'f', 'm', 'm', 'f', 'm', 'f', 'm', 'f', 'm'],
     'employed': [1,1,0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0],
     'income': [45000, 62000, 37000, 82000, 123000, 134000, 156000, 262000, 410000, 25000, 46320, 19000, np.nan]
     }
df = pd.DataFrame(data=d)
print(df)

# Duplicate values analysis
a = ['a', 13, 'a', 1, 'a', 9, 1]
b = ['b', 'a', 'b', 4, 'b', 2, 4]
c = ['c', 7, 'c', 1, 'c', 8, 1]
d = ['d', 2, 'd', 0, 'd', 5, 0]
df = pd.DataFrame(dict(var_1 = a, var_2 = b, var_3 = c, var_4 = d))

# ------------------------------------------------
# 'False' reveals how many registries have X (we do not know the value of X) duplicated values in the dataset?
# Or the same, how many registries in the data set have at least one duplicate in the data set?
df.duplicated(keep = False).value_counts()
# ------------------------------------------------
# ------------------------------------------------
# True reveals the number of duplicated registries in the data set.
# False reveals the number of unique registries in the data set.
df.duplicated(keep = 'first').value_counts()
#-------------------------------------------------
# Number of appearances of each registry
df.groupby(df.columns.tolist(),as_index=False).size()



df.dtypes


#regex inicio y fin de strings


re.match('^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$', '2017-11-30 23:59:58')
re.match('^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$', '2017-11-30 23:59:58,0069')

# Create vertical bar plot: factorplot:
h = sns.factorplot("variable","complete_metric", data=df_completeness_dqss,kind="bar",palette="Blues",size=6,aspect=2,legend_out=False)
h.set_xticklabels(rotation=90)


# Example of IF
if (1 % 1000 == 0): 
	print("Row number loaded: ", i, flush = True)

# Access to the cell of a data frame
df.aloc[i][j]
df.aloc[i]["5 years"]

# Find and replace in Sublime Text 3
Alt+Cmd+F

# Obtain all objects saved in memory
dir()

# retrieve number of rows
len(df)
# retrieve number of columns
len(df.columns)

#Forma de hacer print fácil:
print("Converting numerical variable: " + col, flush = True)

# Creation of toy dataset
df2 = pd.DataFrame(np.random.randint(low=0, high=10, size=(5, 5)),columns=['a', 'b', 'c', 'd', 'e'])
# Acces a column in python dataframe
df[['issue_d']] # This one retrieves data frame
df['issue_d'] # This one retrieves Series

# data
# See the data table in all its width:
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# data

# 6. Order columns in alphabetical order:
data = data.reindex_axis(sorted(data.columns), axis=1)
data.head(4)
# data.shape
data.describe()  # (16964, 129)

#regular expressions
import re
reg = re.compile("[a-z]+8?")
str = "ccc8"
print(reg.match(str).group())

## regex
re.match('^[0-9]{5}/[0-9]{5}$', 'dfwfe4%4') == None

# 7. Pre-Processing
missing_values_table(data)
# 7.1. Drop columns with more than 51% of missings
data = data.dropna(axis=1, thresh=0.51*data.shape[1])
# data.shape
# 7.2. Drop not predictive a priori variables
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
data = data.drop(non_predictive, 1)  # (16964, 94)
# data.shape
non_predictive_comp = ["COEF_DISPERSIO",
                       "VALOR_TOTAL_MAXIM",
                       "VALOR_MOSTRA",
                       "VALOR_MOSTRA_PROJECTAT_ANY0",
                       "VALOR_MOSTRA_PROJECTAT_ANY1",
                       "VALOR_MOSTRA_PROJECTAT_ANY2",
                       "VALOR_MOSTRA_PROJECTAT_ANY3",
                       "VALOR_MOSTRA_PROJECTAT_ANY4",
                       "COEF_CADASTRAL_MOSTRA",
                       "RM_MOSTRA"]
data = data.drop(non_predictive_comp, 1)  # (16964, 84)
contextual_info = ["NUM_JUSTIFICANT_AUTO",
                   "ID_SOLICITUD_VALO",
                   "DADES_PROVISIONALS_CADASTRE",
                   "DADES_PROVISIONALS_QUADRE",
                   "ID_EXPEDIENT",
                   "DATA_DOCUMENT",
                   "ID_OFICINA_COMPETENT",
                   "REF_CADASTRAL",
                   "BLOC",
                   "CARRER",
                   "ESCALA",
                   "KILOMETRO",
                   "PIS",
                   "PORTA",
                   "PRIMER_NUM",
                   "TIPU_VIA",
                   "REF_CATASTRAL",
                   "PARCELLA",
                   "NUM_SECUENCIAL_BIEN",
                   "CARREG",
                   "PCAT1",
                   "COD_INE",
                   "COD_DGC",
                   "COD_POSTAL",
                   "ID_MUNICIPI2",
                   "COORD_X",
                   "COORD_Y",
                   "COD_INTERN_BIEN_CADASTRAL",
                   "COD_INTER_BE_CADASTRAL",
                   "UNITAT_SUPERFICIE",
                   "EXERCICIO",
                   "CARACT_CONTROL_UNO",
                   "CARACT_CONTROL_DOS",
                   "TIPOLOGIA_CADASTRE"
]
data = data.drop(contextual_info, 1)  # (16964, 50)

data.dtypes
import numpy as np
df_dtypes = np.array(data.dtypes)
data.describe(include="all")
df.value_counts()

# Nulls in dataset (not equal)
pd.isnull(df.loc[0, 'B'])

# Access to a column:
df['issue_d']

Skip to content
This repository
Search
Pull requests
Issues
Marketplace
Explore
 @quantumds
Sign out
1
0 0 quantumds/utilities Private
 Code  Issues 0  Pull requests 0  Projects 0  Wiki  Insights  Settings
utilities/python/X_FORMULARY.py
60dfe6c  6 days ago
@quantumds quantumds Update X_FORMULARY.py
     
590 lines (464 sloc)  18.7 KB
# frequency table for categorical or boolean type
plms['isfiller'].value_counts()

# change a column to type string
df['id']= df['id'].astype(str)

# 1. Completeness
# PLMS
# First we create the dataframes with the information:
completeness_plms = 100 - plms.isnull().sum()/len(plms)*100
completeness_dqss = 100 - dqss.isnull().sum()/len(dqss)*100
# Convert completeness_plms into data frames:
x = list(completeness_plms.index)
y = completeness_plms
df_completeness_plms = pd.DataFrame(dict(x=x, y=y))
df_completeness_plms.columns = ['variable', 'complete_metric']
# Create vertical bar plot: factorplot:
g = sns.factorplot("variable","complete_metric", data=df_completeness_plms,kind="bar",palette="Blues",size=6,aspect=2,legend_out=False)
g.set_xticklabels(rotation=90)

# Create horizontal bar plot: barplot
f, ax = plt.subplots(figsize=(18, 25)) # Set size of the graph
sns.set(font_scale = 2) # Set size of the variable labels in each axis, as well as the axis values
sns.barplot(x = "complete_metric", y = "variable", data = df_completeness_dqss, label="Total", palette=sns.color_palette("husl", 20)) # Create the barplot
plt.xlabel("Completeness") # Assign name of the x label
plt.ylabel("Variables") # Assign name of the y label
plt.title("Good results in 'Completeness' metric") # Assign title of the graph
plt.show(sns.barplot) # Show everything

# blue horizontal bar plot
f, ax = plt.subplots(figsize=(18, 25)) # Set size of the graph
sns.set(font_scale = 2) # Set size of the variable labels in each axis, as well as the axis values
sns.barplot(x = "complete_metric", y = "variable", data = df_completeness_dqss, label="Total", color='b') # Create the barplot
plt.xlabel("Completeness") # Assign name of the x label
plt.ylabel("Variables") # Assign name of the y label
plt.title("Good results in DQSS 'Completeness' metric") # Assign title of the graph

# End a process in terminal
# Click in terminal and write: "Ctrl + C"
# See which is our current directory 
os.getcwd()
# Change directory:
os.chdir(route_of_new_directory)







# See names of columns
list(df)
# Assign names to rows index
df.index = df['id']
# Select categorical variables / select variables of type category / Names of columns with type Category:
list(df.select_dtypes(['category']))

# number of levels of categorical variables
df.select_dtypes(['object']).apply(lambda x: len(set(x)))
df.select_dtypes(['category']).apply(lambda x: len(set(x)))
df.select_dtypes(['object']).apply(lambda x: len(set(x)))
df.select_dtypes(['bool']).apply(lambda x: len(set(x)))


# change data type, convert type to string
df['id']= df['id'].astype(str)
# change data type, convert type to string
df['id']= df['id'].astype('str')

# Change types using a loop
for col in ['parks', 'playgrounds', 'sports', 'roading']:
    public[col] = public[col].astype('category')
    
# Frequency table for any variable    
df['hardship_amount'].value_counts(dropna=False) #It drops Na autmatically
# count number of missings for a column
df['hardship_amount'].isnull().sum()
# show number of missings for each column
df.isnull().sum()
# show number of missings in percentage
df.isnull().sum()/len(df)*100
# check what or which registries are equal to a value
df.loc[df['hardship_amount'] == '536.81']

# Block cooment es para comentar todo lo que escojes
# Ctrl + 4 in spyder

# frequency tables
df[col].value_counts()

#frequency tabls above zero
c = df['title'].value_counts()
c = c[c > 0]

# Convert object variables to category:
df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))

# Assign names to rows index:
df.index = df['id']
# Check:
list(df.index)

# Select all variables that arent several types of columns:
list(df.select_dtypes(exclude=["number","bool_","object_"]))

# select object types of variables:
list(df.loc[:, df.dtypes == object])

# select categorical variables / object types of variables:
list(df.loc[:, df.dtypes == object])

df_plms.packages.quantile(0.0) # 10th percentile
df_plms.packages.quantile(0.1) # 10th percentile
df_plms.packages.quantile(0.2) # 10th percentile
df_plms.packages.quantile(0.3) # 10th percentile
df_plms.packages.quantile(0.4) # 10th percentile
df_plms.packages.quantile(0.5) # 10th percentile
df_plms.packages.quantile(0.6) # 10th percentile
df_plms.packages.quantile(0.7) # 10th percentile
df_plms.packages.quantile(0.8) # 10th percentile
df_plms.packages.quantile(0.9) # 10th percentile
df_plms.packages.quantile(1.0) # 10th percentile

# Assign names to rows index:
df.index = df['id']

# Watch every columns data type:
df.dtypes
# Convert all columns to string:
df = df.astype(str)
#Obtain column names / names of columns
list(df.columns.values)
list(df)
# Watch row names
list(df.index)

# Assign names to rows
df.index['Row 2':'Row 5'] 
# See names of columns
list(df)
# Assign names to rows index
df.index = df['id']
 
# Example of FOR
for j in range(12, 25):
	df.iloc[i][j] = df.iloc[i][j+1]
# Interrupt Python console:
Ctrl + C
    
# Introduce an na in python:
np.nan

# % of missings in each column. The values are in X/100 not X/1 
df.isnull().sum()/len(df)*100

# nan is the same as NaN and as NAN

# "palette" argument can be interchanged by "color" argument in library seaborn
# "palette" argument in barplot (seaborn library) valid values:
'Blues_d'
'muted'  
'RdBu'
'Set1'
n = number_of_different_colors_you_want_in_your_barplot
sns.color_palette("husl", n)
# Ex: barplot with 20 bars:
n = 20
sns.color_palette("husl", n)

# Access to the cell of a data frame
list(df.index)
# See names of columns
list(df)

# Other form of creating a data frame: 
x = [1,1000,1001]
y = [200,300,400]
cat = ['first','second','third']
df = pd.DataFrame(dict(x=x, y=y,cat=cat))

# creating factor plots
x = [1,1000,1001]
y = [200,300,400]
cat = ['first','second','third']
df = pd.DataFrame(dict(x=x, y=y,cat=cat))
sns.factorplot("x","y", data=df,kind="bar",palette="Blues",size=6,aspect=2,legend_out=False)

#Create a dataframe in python
    d = {
    'age': [1, 'a', 3, 4, 5, 6, 7, np.nan, np.nan, 10, 1, 1, np.nan], 
     'name': [1, 'a', 3, 4, 5, 6, 6, np.nan, np.nan, 10, 1, 1, np.nan],
     'height': [1, 'a', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10, 1, 1, np.nan],
     'education': [4, 'a', 4, 4, 4, 4, 4,4,4,  4, 1, 1, np.nan],
     'nationality': [4, 'a', 4, 4, 4, 4, 4,np.nan,np.nan,  4, 1, 1, np.nan], 
     'sex': [1, 'a', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10, 1, 1, np.nan],
     'preference': ['a','a','a', 'a', 'a', 'a', 'a', np.nan, np.nan, 'a', 1, 1, np.nan],
     'income': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 1, np.nan],
     'col9': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 1, 1, np.nan],
     'col10': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, np.nan],
     'col11': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, np.nan],
     'col12': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, np.nan],
     'col13': [1,1,1,1,1,1,1,1,1,1, 1, 1, np.nan],
     'col14': ['a','a','a','a','a','a','a','a','a','a', 1, 1,1]
     }
df = pd.DataFrame(data=d)
print(df)

# change object to category
# Change data type to 'category':
plms[plms.select_dtypes(['object']).columns] = plms.select_dtypes(['object']).apply(lambda x: x.astype('category'))
plms.dtypes


#Create a dataframe in python
    d = {
    'age': [32, 41, 35, 43, 57, 63, 73, np.nan, np.nan, 20, 45, 19, np.nan], 
     'name': [1, 'a', 3, 4, 5, 6, 6, np.nan, np.nan, 10, 1, 1, np.nan],
     'height': [1, 'a', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10, 1, 1, np.nan],
     'education': ['law', 'engineering', 'arts', 'economics', 'medicine', 'architecture', 'literature', 'photography', 'cuisine', 'law', 'business_mgmt', 'sport', 'psychology'],
     'nationality': ['india', 'colombia', 'usa', 'sweden', 'italy', 'denmark', 'canada', 'luxembourg', 'england', np.nan, 'south korea', 'australia', np.nan], 
     'sex': ['m', 'm', 'f', 'f', 'f', 'm', 'm', 'f', 'm', 'f', 'm', 'f', 'm'],
     'employed': [1,1,0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0],
     'income': [45000, 62000, 37000, 82000, 123000, 134000, 156000, 262000, 410000, 25000, 46320, 19000, np.nan]
     }
df = pd.DataFrame(data=d)
print(df)

# Duplicate values analysis
a = ['a', 13, 'a', 1, 'a', 9, 1]
b = ['b', 'a', 'b', 4, 'b', 2, 4]
c = ['c', 7, 'c', 1, 'c', 8, 1]
d = ['d', 2, 'd', 0, 'd', 5, 0]
df = pd.DataFrame(dict(var_1 = a, var_2 = b, var_3 = c, var_4 = d))

# ------------------------------------------------
# 'False' reveals how many registries have X (we do not know the value of X) duplicated values in the dataset?
# Or the same, how many registries in the data set have at least one duplicate in the data set?
df.duplicated(keep = False).value_counts()
# ------------------------------------------------
# ------------------------------------------------
# True reveals the number of duplicated registries in the data set.
# False reveals the number of unique registries in the data set.
df.duplicated(keep = 'first').value_counts()
#-------------------------------------------------
# Number of appearances of each registry
df.groupby(df.columns.tolist(),as_index=False).size()

# drop several columns 
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

df.dtypes


#regex inicio y fin de strings


re.match('^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$', '2017-11-30 23:59:58')
re.match('^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$', '2017-11-30 23:59:58,0069')

# Create vertical bar plot: factorplot:
h = sns.factorplot("variable","complete_metric", data=df_completeness_dqss,kind="bar",palette="Blues",size=6,aspect=2,legend_out=False)
h.set_xticklabels(rotation=90)


# Example of IF
if (1 % 1000 == 0): 
	print("Row number loaded: ", i, flush = True)

# Access to the cell of a data frame
df.aloc[i][j]
df.aloc[i]["5 years"]

# Find and replace in Sublime Text 3
Alt+Cmd+F

# Obtain all objects saved in memory
dir()

# retrieve number of rows
len(df)
# retrieve number of columns
len(df.columns)

#Forma de hacer print fácil:
print("Converting numerical variable: " + col, flush = True)

# Creation of toy dataset
df2 = pd.DataFrame(np.random.randint(low=0, high=10, size=(5, 5)),columns=['a', 'b', 'c', 'd', 'e'])
# Acces a column in python dataframe
df[['issue_d']] # This one retrieves data frame
df['issue_d'] # This one retrieves Series

# data
# See the data table in all its width:
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# data

# 6. Order columns in alphabetical order:
data = data.reindex_axis(sorted(data.columns), axis=1)
data.head(4)
# data.shape
data.describe()  # (16964, 129)

#regular expressions
import re
reg = re.compile("[a-z]+8?")
str = "ccc8"
print(reg.match(str).group())

## regex
re.match('^[0-9]{5}/[0-9]{5}$', 'dfwfe4%4') == None

# 7. Pre-Processing
missing_values_table(data)
# 7.1. Drop columns with more than 51% of missings
data = data.dropna(axis=1, thresh=0.51*data.shape[1])
# data.shape
# 7.2. Drop not predictive a priori variables
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
data = data.drop(non_predictive, 1)  # (16964, 94)
# data.shape
non_predictive_comp = ["COEF_DISPERSIO",
                       "VALOR_TOTAL_MAXIM",
                       "VALOR_MOSTRA",
                       "VALOR_MOSTRA_PROJECTAT_ANY0",
                       "VALOR_MOSTRA_PROJECTAT_ANY1",
                       "VALOR_MOSTRA_PROJECTAT_ANY2",
                       "VALOR_MOSTRA_PROJECTAT_ANY3",
                       "VALOR_MOSTRA_PROJECTAT_ANY4",
                       "COEF_CADASTRAL_MOSTRA",
                       "RM_MOSTRA"]
data = data.drop(non_predictive_comp, 1)  # (16964, 84)
contextual_info = ["NUM_JUSTIFICANT_AUTO",
                   "ID_SOLICITUD_VALO",
                   "DADES_PROVISIONALS_CADASTRE",
                   "DADES_PROVISIONALS_QUADRE",
                   "ID_EXPEDIENT",
                   "DATA_DOCUMENT",
                   "ID_OFICINA_COMPETENT",
                   "REF_CADASTRAL",
                   "BLOC",
                   "CARRER",
                   "ESCALA",
                   "KILOMETRO",
                   "PIS",
                   "PORTA",
                   "PRIMER_NUM",
                   "TIPU_VIA",
                   "REF_CATASTRAL",
                   "PARCELLA",
                   "NUM_SECUENCIAL_BIEN",
                   "CARREG",
                   "PCAT1",
                   "COD_INE",
                   "COD_DGC",
                   "COD_POSTAL",
                   "ID_MUNICIPI2",
                   "COORD_X",
                   "COORD_Y",
                   "COD_INTERN_BIEN_CADASTRAL",
                   "COD_INTER_BE_CADASTRAL",
                   "UNITAT_SUPERFICIE",
                   "EXERCICIO",
                   "CARACT_CONTROL_UNO",
                   "CARACT_CONTROL_DOS",
                   "TIPOLOGIA_CADASTRE"
]
data = data.drop(contextual_info, 1)  # (16964, 50)

data.dtypes
import numpy as np
df_dtypes = np.array(data.dtypes)
data.describe(include="all")
df.value_counts()

# Nulls in dataset (not equal)
pd.isnull(df.loc[0, 'B'])

# Access to a column:
df['issue_d']

# -----------------------------------------------------------------------------

import numpy as np
d = {
    'col1': ['a', 1, 'a', 3, 4, 5, 6, 7, np.nan, np.nan, 10, 1, 1, np.nan], 
     'col2': ['b', 1, 'b', 3, 4, 5, 6, 6, np.nan, np.nan, 10, 1, 1, np.nan],
     'col3': ['c', 1, 'c', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10, 1, 1, np.nan],
     'col4': ['d', 4, 'd', 4, 4, 4, 4, 4,4,4,  4, 1, 1, np.nan],
     'col5': ['e', 4, 'e', 4, 4, 4, 4, 4,np.nan,np.nan,  4, 1, 1, np.nan], 
     'col6': ['f', 1, 'f', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10, 1, 1, np.nan],
     'col7': ['g', 'a','g','a', 'a', 'a', 'a', 'a', np.nan, np.nan, 'a', 1, 1, np.nan],
     'col8': ['h', np.nan, 'h', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 1, np.nan],
     'col9': ['i', 'a', 'i', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a','a', 'a'],
     'col10': ['j', 10, 'j', 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, np.nan],
     'col11': [7, 10, 'k', 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 5],
     'col12': [6, 10, 7, 8, 7, 6, 5, 4, 3, 2, 7, 3, 4, 5],
     'col13': ['m', 1,'m',1,1,1,1,1,1,1,1, 1, 1, np.nan],
     'col14': ['a', 'b','c','d','e','f','g','h','i','k','k', 'l', 'm','n']
     }
test = pd.DataFrame(data=d)
print(test)

for ax in plt.gcf().axes:
    l = ax.get_xlabel()
    ax.set_xlabel(l, fontsize=1)    
sns.set_color_codes("pastel")
sns.barplot(x = "col12", y = "col14", data = test,
            label="Total", color="b")

ax.plot(x, y, marker='s', linestyle='none', label='small')

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 24), ylabel="",
       xlabel="Automobile collisions per billion miles")
sns.despine(left=True, bottom=True)

completeness
completeness[1]
completeness[0]
list(completeness.index)[1]

# ----------------------------------------------------------------------------------------

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))

# Load the example car crash dataset
crashes = sns.load_dataset("car_crashes").sort_values("total", ascending=False)



datetime.datetime.strptime(df[['issue_d']],'%b-%Y').strftime('%Y%m%d')



df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")

df["issue_d"]
df[["issue_d"]]



plt.plot(df["event"], df["value_1"])
plt.gcf().autofmt_xdate()
plt.show()



%d - 2 digit date

%b - 3-letter month abbreviation

%Y - 4 digit year

%m - 2 digit month




# Opcion 1:
import dfgui
dfgui.show(df)
# Necesito trabajar en debugging. Source: 

# Opcion 2:
import pandas as pd
pd.set_option('expand_frame_repr', True)

# Opcion 3:
import webbrowser
import pandas as pd
from tempfile import NamedTemporaryFile
def df_window(df):
    with NamedTemporaryFile(delete=False, suffix='.html') as f:
        df.to_html(f)
    webbrowser.open(f.name)
df_window(df)

# Opcion 4:
import pandas as pd
df = pd.read_csv(file_dir + file_name, sep=',', header = 0, encoding = 'latin-1')
self.datatable = QtGui.QTableWidget(parent=self)
self.datatable.setColumnCount(len(df.columns))
self.datatable.setRowCount(len(df.index))
for i in range(len(df.index)):
    for j in range(len(df.columns)):
        self.datatable.setItem(i,j,QtGui.QTableWidgetItem(str(df.iget_value(i, j))))




# Plot the total crashes
sns.set_color_codes("pastel")
sns.barplot(x="total", y="abbrev", data=crashes,
            label="Total", color="b")

# =============================================================================
# # Plot the crashes where alcohol was involved
# sns.set_color_codes("muted")
# sns.barplot(x="alcohol", y="abbrev", data=crashes,
#             label="Alcohol-involved", color="b")
# =============================================================================

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 24), ylabel="",
       xlabel="Automobile collisions per billion miles")
sns.despine(left=True, bottom=True)

completeness
completeness[1]
completeness[0]
list(completeness.index)[1]


d1 = plms.iloc[: , 0:5769495].values.astype(np.str)
d1 = plms.iloc[: , 0:int(((len(plms))/8))].values.astype(np.str)


X.values[:, 3] = labelencoder_X.fit_transform(X.values[:, 3])


from dask import dataframe as dd
sd = dd.from_pandas(plms, npartitions=3)
print (sd)
dd.DataFrame<from_pa..., npartitions=2, divisions=(0, 1, 2)>
EDIT:

I find solution:

import pandas as pd
import dask.dataframe as dd
from dask.dataframe.utils import make_meta

df = pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})
dsk = {('x', 0): df}

meta = make_meta({'a': 'i8', 'b': 'i8'}, index=pd.Index([], 'i8'))
d = dd.DataFrame(dsk, name='x', meta=meta, divisions=[0, 1, 2])
print (d)
dd.DataFrame<x, npartitions=2, divisions=(0, 1, 2)>











 plms.iloc[:, 0:4].values
plms.iloc[:, 4]
X=dataset.iloc[:, 0:4]

d2 = data[:, n/2:].astype('float')

data = np.hstack(d1, d2)

len(plms.columns)
len(plms)


© 2018 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
API
Training
Shop
Blog
About
Press h to open a hovercard with more details.






























































































# Convert dates in format %b-%Y to numeric: 
def date_to_numeric(df):
  for col in dates:
    print("Converting date variables to years: " + col, flush = True)
    df[col] = df[col].apply(pd.to_datetime, format = '%b-%Y', errors = 'coerce')     
    df[col] = 2018 - df[col].apply(pd.to_datetime, errors = 'coerce').dt.year
  return df

# SmavaRepairDataSet: REPAIR SMAVA DATA SET
def SmavaRepairDataSet(df):
  print('Beginning of Dataframe reparation:')
  for i in range(1, len(df)):
    if (i % 10000 == 0):
      print('Row number loaded: ', i)
    elif ((df.iloc[i]['emp_length'] != '< 1 year') & (df.iloc[i]['emp_length'] != '1 year') & (df.iloc[i]['emp_length'] != '2 years') & (df.iloc[i]['emp_length'] != '3 years') & (df.iloc[i]['emp_length'] != '4 years') & (df.iloc[i]['emp_length'] != '5 years') & (df.iloc[i]['emp_length'] != '6 years') & (df.iloc[i]['emp_length'] != '7 years') & (df.iloc[i]['emp_length'] != '8 years') & (df.iloc[i]['emp_length'] != '9 years') & (df.iloc[i]['emp_length'] != '10+ years') & (df.iloc[i]['emp_length'] != 'NA') & (pd.isnull(df.loc[i, 'emp_length']) == False) & (df.iloc[i]['emp_length'] != 'n/a')):
      for j in range(12, 25):
        df.iloc[i][j] = df.iloc[i][j+1]
  return(df)

# Convert to factors:
def to_category(df):
  df[categories] = df[categories].apply(lambda x: x.astype('category'))
  return df

# Convert to numeric: 
def to_numeric(df):
  df[numericals] = df[numericals].apply(pd.to_numeric, errors='coerce')
  return df

SmavaRepairDataSet(df)

df_sales = df_sales.astype(str)

sales = [{'account': 'Jones LLC', 'Jan': 150, 'Feb': 200, 'Mar': 140},
         {'account': 'Alpha Co',  'Jan': 200, 'Feb': 210, 'Mar': 215},
         {'account': 'Blue Inc',  'Jan': 50,  'Feb': 90,  'Mar': 95 }]
df = pd.DataFrame(sales)

sales = {'account': ['Jones LLC', 'Alpha Co', 'Blue Inc'],
         'Jan': [150, 200, 50],
         'Feb': [200, 210, 90],
         'Mar': [140, 215, 95]}
df_sales = pd.DataFrame.from_dict(sales)

SmavaRepairDataSet2(df_sales)

sys.stdout.flush()

def SmavaRepairDataSet2(df):
  print('beginning')
  for i in range(1, len(df)):
    if (1 % 1 == 0):
      print('Row number loaded: ', i)
    elif ((df.iloc[i]['Jan'] != '150') & (df.iloc[i]['Jan'] != '1 year') & (df.iloc[i]['Jan'] != '2 years') & (df.iloc[i]['Jan'] != '3 years') & (df.iloc[i]['Jan'] != '4 years') & (df.iloc[i]['Jan'] != '5 years') & (df.iloc[i]['Jan'] != '6 years') & (df.iloc[i]['Jan'] != '7 years') & (df.iloc[i]['Jan'] != '8 years') & (df.iloc[i]['Jan'] != '9 years') & (df.iloc[i]['Jan'] != '10+ years') & (df.iloc[i]['Jan'] != 'NA') & (df.iloc[i]['Jan'] != 'n/a')):
      for j in range(2, 3):
        df.iloc[i][j] = df.iloc[i][j+1]
  return(df)

# drop_missings: DROP MISSINGS COLUMNS
drop_missings <- function(data) {
  data <- data[,which(colMeans(is.na(data)) < threshold)]
  return(data)
}

# w_variance: DROP COLUMNS WITHOUT VARIANCE
w_variance <- function(data) {
  data <- Filter(function(x)(length(unique(x))>1), data)
  return(data)
}

# no_equal_cols: DROP COLUMNS WHOSE VALUES ARE EQUAL TO ANY OTHER COLUMN:
no_equal_cols <- function(data) {
  uniquelength <- sapply(data,function(x) length(unique(x)))
  isfac <- sapply(data,inherits,"factor")
  data <- data[,!isfac | uniquelength>1]
  return(data)
}

# no_duplicates: ELIMINATE DUPLICATES
no_duplicates <- function(data) {
  data <- data[!duplicated(data), ]
  return(data)
}

# elim_rows_missings: ELIMINATE ROWS FULL OF MISSINGS
elim_rows_missings <- function(data) {
  data <- data[which(rowMeans(is.na(data)) < threshold), ]
  return(data)
}

# glance: COMPLETE SUMMARY OF DATA
glance <- function(data) {
  output <- str(data, list.len=nrow(data))
  return(output)
}

# update_factors: UPDATE FACTOR VARIABLES IN A TABLE
update_factors <- function(data) {
  cat <- sapply(data, is.factor) # Categorical variables search
  A <- function(x) factor(x)
  data[ ,cat] <- data.frame(apply(data[,cat],2, A))
  return(data)
}

# more_1_level: DROP FACTOR VARIABLES WITH ONLY 1 LEVEL 
more_1_level <- function(data) {
  cat <- sapply(data, is.factor)
  w <- Filter(function(x) nlevels(x)>1, data[,cat])
  others <- data[!sapply(data,is.factor)]
  data <- cbind(w,others)
  return(data)
}

# update: UPDATE TABLE
update <- function(data) {
  data <- update_factors(data)
  data <- w_variance(data)
  data <- more_1_level(data)
  return(data)
}



# drop_columns: DROP SEVERAL COLUMNS 
drop_columns <- function(data, cols_to_drop){
  temporal <- names(data) %in% cols_to_drop
  data <- data[!temporal]
  return(data)
}

# mice_imp: MICE MISSING VALUES IMPUTATION
mice_imp <- function(data, num_sub) {
  library(mice)
  gc()
  imp <- mice(data, m=num_sub, seed=1234)
  data <- complete(imp)
  data <-update(data)
  return(data)
}

# mice_imp_cart: MICE CART MISSING VALUES IMPUTATION
mice_imp_cart <- function(data, num_sub) {
  library(mice)
  gc()
  imp <- mice(data, m=num_sub, method = "cart", seed=1234)
  data <- complete(imp)
  data <- update(data)
  return(data)
}

# missforest_imp: MICE MISSING VALUES IMPUTATION
missforest_imp <- function(data) {
  library(missForest)
  gc()
  imp <- missForest(data)
  data <- imp$ximp
  data <-update(data)
  return(data)
}

# corr_plot: CORRELATION PLOT
corr_plot <- function(data) {
  nums <- data[sapply(data,is.numeric)]
  head(nums)
  names(nums)
  ncol(nums)
  library(corrplot)
  matriu_correlacions1 <- cor(na.omit(nums))
  corrplot(matriu_correlacions1, method = "circle",tl.cex = 0.5)
  return(matriu_correlacions1)
}

# sel_num_drop: SELECT NUMERIC TO DROP
sel_num_drop <- function(data){
  df_varimax <- as.data.frame(data)
  for (j in 1:ncol(df_varimax)){
    print("VARIABLE"); 
    print(as.character(colnames(df_varimax[j])));
    for(i in j+1:nrow(df_varimax)){
      if((df_varimax[i,j] > 0.8 | df_varimax[i,j] < -0.8) & (is.na(df_varimax[i,j]) ==FALSE)) {
        print(as.character(row.names(df_varimax)[i]))
        print(df_varimax[i,j])
      } 
    }
    print("")
  }
}

############################################################################
# DIMENSIONALITY REDUCTION
############################################################################
# extract_pred: EXTRACT PREDICTIVE VARIABLES FROM A DATASET <- The parameters id and var_resp need to be entered between double quotes
# I.e: extract_pred(mytable, "CUSTOMER_ID", "PRICE_TARGET")
extract_pred <- function(data, id, var_resp){
  data <- data[,-c(which(colnames(data)==id),which(colnames(data)==var_resp) ) ]
  return(data)
}

# sel_nums: SELECT NUMERIC VARIABLES
sel_nums <- function(data){
  numbers <- sapply(data, is.numeric) 
  numericals <- data[ ,numbers]
  return(numericals)
}

# sel_cats: SELECT CATEGORICAL VARIABLES
sel_cats <- function(data){
  data <- update(data)
  cat <- sapply(data, is.factor) 
  categoricals <- data[ ,cat]
  return(categoricals)
}

# sel_ids: SELECT IDs
sel_ids <- function(data) {
  chars <- sapply(data, is.character)
  ids <- data[ ,chars]
  return(ids)
}

# return_dim: RETURN NUMBER OF COMPONENTS THAT ACHIEVE 80% OF ACCUMULATED VARIANCE
return_dim <- function(eigen) { 
  for(i in 1:nrow(eigen)){
    if (eigen$cumulative.percentage.of.variance[i] < 80){
    } else {
      return(i)
    }
  }
}

# cand_num_drop: PCA
num_cand_drop <- function(data){
  nums <- sel_nums(data)
  print("1. Numerical variables separation finalized. Beginning categorical variables separation:")
  categ <- sel_cats(data)
  print("2. Categorical variables separation finalized. Beginning id's variables separation:")
  ident <- sel_ids(data)
  print("3. ID's variables separation finalized. Beginning variables merging:")
  data <- cbind(nums,categ)
  print("4. Variables merging finalized. Beginning initial PCA:")
  res_initial <- PCA(data, quali.sup=(ncol(nums)+1):(ncol(nums)+ncol(categ)), scale.unit = positive, graph=positive)
  print("5. Initial PCA finalized. Calculating optimal dimmensions number:")
  eigen <- data.frame(res_initial$eig)
  write.table(eigen, file = file_num_cand, sep = sep_pca, row.names = positive)
  ndim <- return_dim(eigen)
  print("6. Optimal number of dimensions calculated. Beginning definitive PCA:")
  res <- PCA(data[!(names(data) %in% names(ident))], quali.sup=(ncol(nums)+1):(ncol(nums)+ncol(categ)), ncp = ndim, scale.unit = positive, graph=positive)
  print("7. Definitive PCA finalized. Beginning results printing:")
  table_analysis <- data.frame(res$var$cor)
  write.table(table_analysis, file = dir_pca, sep = sep_pca, row.names = positive)
  print("8. Results printing finalized.")
  return(table_analysis)
}

# sel_num_drop_loose: SELECT NUMERIC TO DROP <- THIS IS A MUCH MORE LOOSER CRITERIA <- THRESHOLD IS >0.8, <-0.8 OR SIMPLY MAX, MIN +-0.1
sel_num_drop_loose <- function(df_varimax){
  selected <- df_varimax
  for (j in 2:ncol(df_varimax)){
    for(i in 1:nrow(df_varimax)){
      if(df_varimax[i,j] < -0.8 | df_varimax[i,j] > 0.8){
        selected[i,j] <- df_varimax[i,j]
        print(as.character(colnames(df_varimax[j])))
        print(paste(as.character(df_varimax[i,1]), df_varimax[i,j],sep=" "))
      } else {
        if (df_varimax[i,j] > max(abs(df_varimax[,j]))-0.1){
          selected[i,j] <- max(abs(df_varimax[,j]))
          print(as.character(colnames(df_varimax[j])))
          print(paste(as.character(df_varimax[i,1]), selected[i,j],sep=" "))
        } else if (df_varimax[i,j] != max(abs(df_varimax[,j]))) {
          selected[i,j] <- 0
          print(as.character(colnames(df_varimax[j])))
          print(paste(as.character(df_varimax[i,1]), selected[i,j],sep=" "))
        }
      }
    }
  }
  return(selected)
}

# return_dim_mix: RETURN NUMBER OF COMPONENTS THAT ACHIEVE 80% OF ACCUMULATED VARIANCE
return_dim_mix <- function(eigen) { 
  for(i in 1:nrow(eigen)){
    if (eigen$Cumulative[i] < 80){
    } else {
      return(i)
    }
  }
}

# cat_cand_drop: MCA through PCAmix function
cat_cand_drop <- function(data){
  nums <- sel_nums(data)
  print("1. Numerical variables separation finalized. Beginning categorical variables separation:")
  categ <- sel_cats(data)
  print("2. Categorical variables separation finalized. Beginning id's variables separation:")
  ident <- sel_ids(data)
  print("3. ID's variables separation finalized. Beginning variables merging:")
  data <- cbind(nums,categ,ident)
  print("4. Variables merging finalized. Beginning initial PCAmix:")
  res_inicial <- PCAmix(data[,1:(ncol(nums))],data[,(ncol(nums)+1):(ncol(nums)+ncol(categ))],graph=positive,rename.level=positive)
  print("5. Initial PCAmix finalized. Calculating optimal dimmensions number:")
  eigen <- data.frame(res_inicial$eig)
  write.table(eigen, file = dir_eigen_mix, sep = sep_pca, row.names = positive)
  ndimensions <- return_dim_mix(eigen)
  print("6. Optimal number of dimensions calculated. Beginning definitive PCAmix:")
  res <- PCAmix(data[,1:(ncol(nums))], data[,(ncol(nums)+1):(ncol(nums)+ncol(categ))], ndim = ndimensions, graph=positive,rename.level=positive)
  print("7. Definitive PCAmix finalized. Beginning results printing:")
  table_analysis_pcamix <- data.frame(res$sqload)
  write.table(table_analysis_pcamix, file = dir_pca_mix, sep = sep_pca, row.names = positive)
  print("8. Results printing finalized.")
  return(table_analysis_pcamix)
}

# cat_cand_drop: MCA through PCAmix function
sel_cat_mix_drop <- function(df_varimax_mix){
  selected <- df_varimax_mix
  for (j in 2:ncol(df_varimax_mix)){
    for(i in 1:nrow(df_varimax_mix)){
      if(df_varimax_mix[i,j] == max(df_varimax_mix[,j])){
        selected[i,j] <- df_varimax_mix[i,j]
        print(as.character(colnames(df_varimax_mix[j])))
        print(paste(as.character(df_varimax_mix[i,1]), df_varimax_mix[i,j],sep=" "))
      } else {
        selected[i,j] <- 0
      }
    }
  }
  new <- selected[rowSums(selected[,colnames(selected)[(3:ncol(selected))]]==0)==ncol(selected)-2, ]
  write.table(new, file = dir_selected_mix, sep = sep_pca, row.names = positive)
  return(new)
}

############################################################################
# DIMENSIONALITY REDUCTION
############################################################################
# extract_psi: Extraction of psi for numerical variables
extract_psi <- function(data){
  nums <- sel_nums(data)
  print("1. Numerical variables separation finalized. Beginning categorical variables separation:")
  categ <- sel_cats(data)
  print("2. Categorical variables separation finalized. Beginning id's variables separation:")
  ident <- sel_ids(data)
  print("3. ID's variables separation finalized. Beginning variables merging:")
  data <- cbind(nums,categ)
  print("4. Variables merging finalized. Beginning initial PCA:")
  res_initial <- PCA(data, quali.sup=(ncol(nums)+1):(ncol(nums)+ncol(categ)), scale.unit = positive, graph=positive)
  print("5. Initial PCA finalized. Calculating optimal dimmensions number:")
  eigen <- data.frame(res_initial$eig)
  ndim <- return_dim(eigen)
  print("6. Optimal number of dimensions calculated. Beginning definitive PCA:")
  res <- PCA(data[!(names(data) %in% names(ident))], quali.sup=(ncol(nums)+1):(ncol(nums)+ncol(categ)), ncp = ndim, scale.unit = positive, graph=positive)
  print("7. Definitive PCA finalized. Beginning results printing:")
  res.rot <- varimax(res$var$cor[,1:ndim])
  print("8. Calculation of rotated coordenates:")
  Psi <- res$ind$coord[,1:ndim]       # Psi contains the original most important major components (not rotated)
  Phi <- res$var$coord[,1:ndim]       # Phi contains the correlations of the variables with the main components not rotated. 
  X   <- nums                         # X contains the active data
  p <- ncol(X)                        # p contains the number of active variables
  Xs <- scale(X)                      # Xs contains the standardized data
  iden <- row.names(X)                # iden = row identifiers
  labels_data <- names(X)             # variables labels
  Phi.rot <- res.rot$loadings[1:p,]   # Phi.rot contiene las correlaciones de las variables con las componentes principales rotadas  
  lmb.rot <- diag(t(res.rot$loadings) %*% res.rot$loadings)
  Psi_stan.rot <- Xs %*% solve(cor(X)) %*% Phi.rot
  Psi.rot <- Psi_stan.rot %*% diag(sqrt(lmb.rot)) #Numerical rotated components
  print("9. Finalized extraction of Psi.")
  return(Psi)
}

# extract_psi_cat: Extraction of psi for categorical variables
extract_psi_cat <- function(data){
  print("1. Beginning categorical variables separation:")
  cats <- data[sapply(data,is.factor)] 
  print("2. Categorical variables separation finalized. Beginning initial MCA:")
  res.cat <- MCA(cats, graph=T) 
  print("3. Initial MCA finalized. Beginning optimal dimension extraction:")
  eigen <- data.frame(res.cat$eig)
  ndim <- return_dim(eigen)
  print("3. Optimal dimension extraction finalized. Beginning definitive MCA:")
  res_cat <- MCA(cats, ncp= ndim, graph=T) 
  print("4. MCA finalized. Printing psi_cat:")
  psi_cat <- res_cat$ind$coord[,1:ndim]
  print("5. Printing psi_cat finalized.")
  return(psi_cat)
}






















#____________________________________________________________________________________________________________________________________________________________











categories = [
  'term', 
  'grade',
  'sub_grade', 
  'emp_title',
  'emp_length',
  'home_ownership',
  'verification_status', 
  'pymnt_plan', 
  'purpose', 
  'title',
  'zip_code',
  'addr_state', 
  'initial_list_status', 
  'application_type', 
  'hardship_flag', 
  'l_state'
]  
numericals = [                                
  'loan_amnt',
  'funded_amnt',
  'funded_amnt_inv',
  'int_rate',
  'installment',
  'annual_inc',
  'dti',
  'delinq_2yrs',
  'inq_last_6mths',
  'mths_since_last_delinq',
  'open_acc',
  'pub_rec',
  'revol_bal',
  'revol_util',
  'total_acc',
  'out_prncp',
  'out_prncp_inv',
  'total_pymnt',
  'total_pymnt_inv',
  'total_rec_prncp',
  'total_rec_int',
  'total_rec_late_fee',
  'recoveries',
  'collection_recovery_fee',
  'last_pymnt_amnt',
  'collections_12_mths_ex_med',
  'acc_now_delinq', 
  'tot_coll_amt',
  'tot_cur_bal',
  'open_acc_6m',
  'open_il_6m',
  'open_il_12m',
  'open_il_24m',
  'mths_since_rcnt_il',
  'total_bal_il',
  'il_util',
  'open_rv_12m',
  'open_rv_24m',
  'max_bal_bc',
  'all_util',
  'total_rev_hi_lim',
  'inq_fi',
  'total_cu_tl',
  'inq_last_12m',
  'acc_open_past_24mths',
  'avg_cur_bal',
  'bc_open_to_buy',
  'bc_util',
  'chargeoff_within_12_mths',
  'delinq_amnt',
  'mo_sin_old_il_acct',
  'mo_sin_old_rev_tl_op',
  'mo_sin_rcnt_rev_tl_op',
  'mo_sin_rcnt_tl',
  'mort_acc',
  'mths_since_recent_bc',
  'mths_since_recent_inq',
  'num_accts_ever_120_pd',
  'num_actv_bc_tl',
  'num_actv_rev_tl',
  'num_bc_sats',
  'num_bc_tl',
  'num_il_tl',
  'num_op_rev_tl',
  'num_rev_accts',
  'num_rev_tl_bal_gt_0',
  'num_sats',
  'num_tl_120dpd_2m',
  'num_tl_30dpd',
  'num_tl_90g_dpd_24m',
  'num_tl_op_past_12m',
  'pct_tl_nvr_dlq',
  'percent_bc_gt_75',
  'pub_rec_bankruptcies',
  'tax_liens',
  'tot_hi_cred_lim',
  'total_bal_ex_mort',
  'total_bc_limit',
  'total_il_high_credit_limit'
]
dates = [
  'issue_d',
  'earliest_cr_line',
  'last_pymnt_d',
  'next_pymnt_d',
  'last_credit_pull_d'
]
file_name_test = 'RAcredit_test.csv'
sep_char = ''
thresh_cols = 0.6 # 60%
thresh_rows = 0.6 # 60%
addr_state_values = [
'CA',
'TX',
'NY',
'FL',
'IL',
'NJ',
'PA',
'OH',
'GA',
'VA',
'NC',
'MI',
'MD',
'MA',
'AZ',
'CO',
'WA',
'MN',
'IN',
'CT',
'TN',
'MO',
'NV',
'WI',
'SC',
'AL',
'LA',
'OR',
'KY',
'OK',
'KS',
'AR',
'UT',
'MS',
'NE',
'NM',
'NH',
'RI',
'HI',
'DE',
'ME',
'MT',
'ID',
'AK',
'ND',
'VT',
'DC',
'SD',
'WY'
]






















#___________________________________________________________________________________________________________________________________________________________



# Rule 6: 'eventcode'
plms.duplicated('eventcode', keep = 'first').value_counts() # 986 repeated values
repeated_eventcode = plms[plms.duplicated('eventcode', keep = 'first') == False]
# Rule 7: 'replacedeventcode'

# -----------------------------------------------------



# Let's just make a 1-by-2 plot
df = df.head(10)

# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(df, col="walk", hue="walk", col_wrap=2, size=5,
        aspect=1)

# Draw a bar plot to show the trajectory of each random walk
bp = grid.map(sns.barplot, "step", "position", palette="Set3")

# The color cycles are going to all the same, doesn't matter which axes we use
Ax = bp.axes[0]

# Some how for a plot of 5 bars, there are 6 patches, what is the 6th one?
Boxes = [item for item in Ax.get_children()
         if isinstance(item, matplotlib.patches.Rectangle)][:-1]

# There is no labels, need to define the labels
legend_labels  = ['a', 'b', 'c', 'd', 'e']

# Create the legend patches
legend_patches = [matplotlib.patches.Patch(color=C, label=L) for
                  C, L in zip([item.get_facecolor() for item in Boxes],
                              legend_labels)]

# Plot the legend
plt.legend(handles=legend_patches)

# Convention done after re-coding
df_completness -> df_completeness_plms

# 3. Consistency

# 4. Conformity

# 5. Currency

# 6. Duplication

# 7. Integrity

############################################################################
# DATA QUALITY 
############################################################################
# 1. Completeness
sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))

# Load the example car crash dataset
crashes = sns.load_dataset("car_crashes").sort_values("total", ascending=False)

# Plot the total crashes
sns.set_color_codes("pastel")
sns.barplot(x="total", y="abbrev", data=crashes,
            label="Total", color="b")

# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
sns.barplot(x="alcohol", y="abbrev", data=crashes,
            label="Alcohol-involved", color="b")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 24), ylabel="",
       xlabel="Automobile collisions per billion miles")
sns.despine(left=True, bottom=True)

############################################################################
# DESCRIPTIVE ANALYTICS
############################################################################
# PLMS
# 'packages'
f, ax = plt.subplots(figsize=(15, 15))
sns.set(font_scale = 2)
sns.distplot((plms['packages'] ))

plms['packages'].describe(percentiles=[.0, .1, .2 , .3, .4, .5, .6, .7, .8, .9, .91, .92, .93, .94, .95, .96, .97, .98, .99, 1])
# WH issue related

f, ax = plt.subplots(figsize=(15, 15))
sns.set(font_scale = 2)
sns.distplot(((plms.loc[plms['packages'] >= 3])['packages'] ))

f, ax = plt.subplots(figsize=(15, 15))
sns.set(font_scale = 2)
sns.distplot(((plms.loc[plms['packages'] >30])['packages'] ))

# iswarning
plms['iswarning'].value_counts()

# isfiller
plms['isfiller'].value_counts()

# Instead of having number of packages, a boolean variable 

# DQSS
# Relation between Sampling Time and Defect Packages
# Setting colors
dqss.plot(x='samplingtime', y='defectpackages', figsize=(19,10))

# Distribution plot of the sampled packages
sns.set(font_scale = 2)
f, ax = plt.subplots(figsize=(15, 15))
sns.distplot(dqss['sampledpackages'])

# Add analysis of:
- type of events
- production flag
- sample type
- one is normal sample and the other is the unit
- normalize the number of packages
- sample deffect column
- was in the laboratory

dqss['samplingtime']

# hypothesis in title of charts

plms.loc[plms['packages'] > 0]

sns.distplot(plms['isfiller'])
sns.distplot(plms['iswarning'])

dqss['evdescreng']
list(dqss)

df.field_A.quantile(0.5) # same as median
# 62.0

df.field_A.quantile(0.9) # 90th percentile

plms.isnull().sum()/len(plms)*100

sns.distplot(dqss['defectpackages'])
american = df['nationality'] == "USA"
# Create variable with TRUE if age is greater than 50
elderly = df['age'] > 50

# Plotting the graph
sns_plot = sns.barplot(x="completeness", y="variable", data = completeness_df, label="Total", color="b")
sns_plot.figure.savefig("output.png")

import pandas as pd
import numpy as np
from pandas import Series, DataFrame, Panel
pd.set_option('display.max_rows',15) # this limit maximum numbers of rows

# Plot using seaborn
sns.set(font_scale = 2)
b = sns.violinplot(y = "Draughts", data = df)
plt.show()

# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
sns.barplot(x="alcohol", y="abbrev", data=crashes,
            label="Alcohol-involved", color="b")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 24), ylabel="",
       xlabel="Automobile collisions per billion miles")
sns.despine(left=True, bottom=True)

############################################################################
# DATA QUALITY ASSESSMENT
############################################################################
# 1. Completeness
# Percentage of fulfilled values per column
completeness = 100 - df.isnull().sum()/len(df)*100
# Conversion of Series to DataFrame
completeness_df = pd.DataFrame({ 'completeness':completeness, 'variable':completeness.index})
completeness_df.reset_index(inplace = True)
completeness_df = completeness_df.drop('index', 1)
# Setting the size of the matplotlib window
f, ax = plt.subplots(figsize=(40, 20)) # The first component is X axis , the second is Y axis
# Setting colors
sns.set_color_codes("pastel")
# Setting size of the tabs in each axis
sns.set(font_scale=4)
# Plotting the graph
sns_plot = sns.barplot(x="completeness", y="variable", data = completeness_df, label="Total", color="b")
sns_plot.figure.savefig("output.png")

# 2. Accuracy
# Accuracy
syn(plms)

plms.dtypes
plms[plms.select_dtypes(['object']).columns] = plms.select_dtypes(['object']).apply(lambda x: x.astype('category'))

############################################################################
# IMPORT DATA
############################################################################
plms = pd.read_csv(file_dir + file_name_plms, sep = ',', header = 0, encoding = 'latin-1', low_memory = False)
dqss = pd.read_csv(file_dir + file_name_dqss, sep = ',', header = 0, encoding = 'latin-1', low_memory = False)

############################################################################
# DATA TYPE ANALYSIS
############################################################################
# Initial inspection
plms.dtypes
syn(plms)
# Data Type Conversions:
# -> string:
plms[to_str] = plms[to_str].astype(str)
# -> category:
plms[to_cat] = plms[to_cat].apply(lambda x: x.astype('category'))

# Detection of Primary Key
plms.duplicated('idperfexp', keep = 'first').value_counts()
plms.duplicated('idline', keep = 'first').value_counts()
plms.duplicated('sn', keep = 'first').value_counts()
plms.duplicated('idequipment', keep = 'first').value_counts()
plms.duplicated('startdatetime', keep = 'first').value_counts()
plms.duplicated('eventcode', keep = 'first').value_counts()
plms.duplicated('eventtext', keep = 'first').value_counts()
plms.duplicated('originaleventcode', keep = 'first').value_counts()
plms.duplicated('replacedeventcode', keep = 'first').value_counts()
plms.duplicated('charval', keep = 'first').value_counts()

############################################################################
# DATA QUALITY ASSESSMENT
############################################################################
# 1. Completeness
# PLMS
# First we create the dataframes with the information:
completeness_plms = 100 - plms.isnull().sum()/len(plms)*100
completeness_dqss = 100 - dqss.isnull().sum()/len(dqss)*100
# Convert completeness_plms into data frames:
x = list(completeness_plms.index)
y = completeness_plms
df_completeness_plms = pd.DataFrame(dict(x=x, y=y))
df_completeness_plms.columns = ['variable', 'complete_metric']
# Create horizontal bar plot: barplot
f, ax = plt.subplots(figsize=(18, 25)) # Set size of the graph
sns.set(font_scale = 2) # Set size of the variable labels in each axis, as well as the axis values
sns.barplot(x = "complete_metric", y = "variable", data = df_completeness_plms, label="Total", color='b') # Create the barplot
plt.xlabel("Completeness") # Assign name of the x label
plt.ylabel("Variables") # Assign name of the y label
plt.title("Good results in PLMS 'Completeness' metric") # Assign title of the graph

# DQSS
# Convert completeness_dqss into data frames:
x = list(completeness_dqss.index)
y = completeness_dqss
df_completeness_dqss = pd.DataFrame(dict(x=x, y=y))
df_completeness_dqss.columns = ['variable', 'complete_metric']
# Create vertical bar plot: factorplot:
h = sns.factorplot("variable","complete_metric", data=df_completeness_dqss,kind="bar",palette="Blues",size=6,aspect=2,legend_out=False)
h.set_xticklabels(rotation=90)
# Create horizontal bar plot: barplot
f, ax = plt.subplots(figsize=(18, 25)) # Set size of the graph
sns.set(font_scale = 2) # Set size of the variable labels in each axis, as well as the axis values
sns.barplot(x = "complete_metric", y = "variable", data = df_completeness_dqss, label="Total", color='b') # Create the barplot
plt.xlabel("Completeness") # Assign name of the x label
plt.ylabel("Variables") # Assign name of the y label
plt.title("Good results in DQSS 'Completeness' metric") # Assign title of the graph

# 2. Accuracy:
# PLMS
plms.dtypes 
syn(plms) # 100% Accuracy

# 3. Consistency:
syn(plms) # 100% Accuracy
# idperfexp
# Is there any letter in the values?
any(x.isalpha() for x in plms['idperfexp'])
any(x.isalnum() for x in  plms['idperfexp'])
any(x.isdigit() for x in  plms['idperfexp'])







plms['idperfexp'].astype(str).str[0:7]




plms['idperfexp':4]


a_string = 'This is a string'
first_four_letters = a_string[:4]

>>> 'This'
Or the last 5:

last_five_letters = a_string[-5:]
>>> 'string'
So applying that logic to your problem:

the_string = '416d76b8811b0ddae2fdad8f4721ddbe|d4f656ee006e248f2f3a8a93a8aec5868788b927|12a5f648928f8e0b5376d2cc07de8e4cbf9f7ccbadb97d898373f85f0a75c47f '
first_32_chars = the_string[:32]
>>> 416d76b8811b0ddae2fdad8f4721ddbe






# 4. Currency:
# No information regarding currency.

# 5. Duplication:
# 'True' reveals the number of duplicated registries in the data set.
plms.duplicated(keep = 'first').value_counts() # There are no duplicates in the entire data set



