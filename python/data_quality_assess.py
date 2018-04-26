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















