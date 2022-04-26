    
# HEAD OF SCRIPTS 
############################################################################
# Title: XXX
# Date: XX de nombre_mes de XXXX
############################################################################

# CHUNK OF CODE SEPARATOR
############################################################################
# ENVIRONMENT SETTINGS
############################################################################

############################################################################
                    # NAME OF PART
############################################################################

# GENERAL KNOWLEDGE
# Single quotes are reserved to characters.
# Double quotes are reserved to strings.
# Julia does not permit operators (+, -, *, ...), "methods" in Julia; that are not previously defined.
# There is no way to clear the workspace in Julia. You need to restart the session
# It seems that Symbol is string but we will confirm it soon

# INSTALL LIBRARIES
using Pkg # "using" keyword must be always in lowerkeys
Pkg.add("CSV")  # CSV is the name of the library

# LENGTH OF A VECTOR OR LENGTH OF A LIST
size(name_of_list)

# WATCH COMPLETE DATA FRAME / SEE WHOLE DATA FRAME / 
showall(df)

# UNIQUE VALUES OF A LIST
unique(name_of_list)

# FUNCTIONS
function name_of_function(parameter 1, parameter2, ..., parametern) # No semicolons or colons, no 2 points, etc.
    operations...
end

# DIMENSION OF A DATA FRAME
size(df)

# FOR LOOPS
for x in list
    operations...
end

# IMPORT JULIA SCRIPT OR JULIA FILE FROM OTHER DIRECTORY
include("path_to_julia_file/name_of_julia_file.jl")

# FILTER A DATA FRAME 
# You cannot use simply the boolean expression alone, you must end indicating all the columns with: " , :" at the end.
df[df[:review_date] .== Date(2019,1,4) , :] # The "equal operator" has a point in front
df[df[:number_of_cars] .== 7 , :] 

# GENERATE A SEQUENCE
collect(1:5) # Te sequence generated is: 1 2 3 4 5

# ORDER / SORT / DATA FRAME DEPENDING ON VALUES OF A COLUMN
sort(df, :name_of_column) # Increasing order depending on the values of name_of_column
sort(df, :name_of_column, rev=true) # Decreasing order depending on the values of name_of_column

# INSTALL PACKAGES
Pkg.update() # It is a good philosophy to update previous to a new installation to avoid problems.
using Pkg # "using" keyword must be always in lowerkeys
Pkg.add("name_of_package")

# LOAD LIBRARIES
using name_of_library

# CREATE A SEQUENCE / GENERATE SEQUENCES
collect(1:8)

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
df_values = CSV.read(data_dir * raw_file_name, header = false, delim = "|")
df_colnames = CSV.read(data_dir * raw_colnames_file_name, header = false, delim = ",")

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

# APPEND ROWS TO A DATA FRAME / ADD ROWS TO A DATA FRAME
# The row should be a data frame; and in this case, if df2 is the row to append, df1 should not be empty
append!(df1, df2)
# This option works with feeding empty data frames:
push!(df1, df2[ number_of_line , :])
# For example:
df1 = DataFrame(A = 1:4, B = ["M", "F", "F", "M"])
df2 = DataFrame(A = 6, B = ["F"])

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

# MEASURE TIME OF A PROCEDURE IN JULIA
# Import Libraries:
using Distributed # Import required library
addprocs(4) # Work with 4 processors

# Measure time with a simple loop:
@time begin # Keyword to note that we are measuring from here
  for N in 1:10000
    println("The N of this iteration in $N")
  end
end # Keyword to say that we end measuring time

# Measure the time with a simple function:
function printings(num) # We create first the function
    for N in 1:num
        println("The N of this iteration in $N")
    end
end
@time printings(5000) # We call the execution of the function and measurement of time simply putting "@time" before
