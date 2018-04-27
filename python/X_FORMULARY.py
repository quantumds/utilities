# frequency table for categorical or boolean type
plms['isfiller'].value_counts()

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
