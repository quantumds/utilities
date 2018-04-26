




























































































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