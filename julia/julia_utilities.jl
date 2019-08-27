# GENERAL KNOWLEDGE
# Single quotes are reserved to characters.
# Double quotes are reserved to strings.
# Julia does not permit operators (+, -, *, ...), "methods" in Julia; that are not previously defined.

# PRINT THROUGH THE SCREEN
println(object_to_print)

# TYPE OF AN OBJECT
typeof(name_of_object)

# CHANGE WORKING DIRECTORY
cd("/Users/amsistac/Documents/ANIBAL/data_science/julia_projects/autos_uci_julia/data") # Mac systems

# SEE / PRINT / VIEW CURRENT WORKING DIRECTORY
pwd()

# READ DATA / READ CSV / READ TABLE / READ DATA FRAME
df_values = CSV.read(data_dir * raw_file_name, header = false)
df_colnames = CSV.read(data_dir * raw_colnames_file_name, header = false)

# CONVERT 1 ROW DATA FRAME TO STRING VECTOR
list_colnames = vec(convert(Array, df_colnames[1, :])) # The 1st row contained the column names
