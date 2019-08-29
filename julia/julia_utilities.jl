    
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

# INSTALL PACKAGES
Using Pkg
Pkg.add("name_of_package")

# LOAD LIBRARIES
Use name_of_library

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
