    
#CABECERA DE SCRIPTS / HEAD DE SCRIPTS /
############################################################################
# Título: XXX
# Fecha: XX de nombre_mes de XXXX
# Tratamiento: Train ó Tes
# Tipo de datos: XX
############################################################################

#SEPARADOR DE TROZOS DE CÓDIGO
############################################################################
# ENVIRONMENT SETTINGS
############################################################################

############################################################################
                    # NOMBRE_DE_LA_PARTE
############################################################################

# GENERAL KNOWLEDGE
# Single quotes are reserved to characters.
# Double quotes are reserved to strings.
# Julia does not permit operators (+, -, *, ...), "methods" in Julia; that are not previously defined.
# There is no way to clear the workspace in Julia. You need to restart the session
# It seems that Symbol is string but we will confirm it soon

# GENERATE A SEQUENCE
collect(1:5) # Te sequence generated is: 1 2 3 4 5

# INSTALL PACKAGES
Pkg.update() # It is a good philosophy to update previous to a new installation to avoid problems.
using Pkg # "using" keyword must be always in lowerkeys
Pkg.add("name_of_package")

# LOAD LIBRARIES
using name_of_library

# PRINT THROUGH THE SCREEN
println(object_to_print)

# TYPE OF AN OBJECT
typeof(name_of_object)

# CHANGE WORKING DIRECTORY
cd("/Users/amsistac/Documents/ANIBAL/data_science/julia_projects/autos_uci_julia/data/") # Mac systems
cd("\\Users\\amsistac\\Documents\\ANIBAL\\data_science\\julia_projects\\autos_uci_julia\\data\\") # Windows uses double backslashes \\

# SEE / PRINT / VIEW CURRENT WORKING DIRECTORY
pwd()

# READ DATA / READ CSV / READ TABLE / READ DATA FRAME
df_values = CSV.read(data_dir * raw_file_name, header = false)
df_colnames = CSV.read(data_dir * raw_colnames_file_name, header = false)

# CONVERT 1 ROW DATA FRAME TO STRING VECTOR
list_colnames = vec(convert(Array, df_colnames[1, :])) # The 1st row contained the column names

# ASSIGN AS COLUMN NAMES 1 ONE STRING VECTOR
names!(df, Symbol.(list_colnames))

# MISSINGS
# Number of total missings in a Data Frame
sum(colwise(x -> sum(ismissing.(x)), df))

# PASTE A SRING
string1 * string2 # operator "*" is like "paste" in Python or R 
# Example
string1 = "Hola. "
string2 = "Cómo estás? "
string1 * string2 # This will give result by console like: "Hola. Cómo estás?"

# DESCRIPTIVE SUMMARY / DESCRIPTIVE STATISTICS
describe(df)

# OPEN PREVIOUS SAVED CONFIGURATION
Step 1: Shift+Super+P
Step 2: Type: "save workspace open"
Step 3: Click in the option of "Open"
Step 4: Double click in julia_main.jl

# ACCESS TO A COLUMN
df[:name_of_column] # 1st way
df.name_of_column # 2nd way
df[:, 5] # 5th column, all rows
df[:, 5:8] # 5th to 8th column, all rows

# CREATING NUMERICAL LISTS
days_vals = [0, 0, 0, 0]

# CREATING LISTS OF DATES
created_date_vals = [Date(2019,1,1), Date(2019,1,1), Date(2019,1,1), Date(2019,1,1)]

# DATES
Date(2019,12,31) # The format is year, Month, Day # Date is the simple format (no miliseconds). Datetime is the complex one

# CREATE A NEW DATA FRAME / DATAFRAME
# One example:
df = DataFrame(a = 1:4, b = 1:4, c = randn(4), d = randn(4))
# Another example:
df = DataFrame(opportunity_number = op_num_vals,
                            created_date = created_date_vals,
                            review_date = rev_date_vals)
