# LOAD PARQUET FILES
df = spark.read.parquet(route_of_your_parquet_file)
# OBTAIN HEADER
df.show(n=4) # n = number of lines that you want to see in the header of the file

# OBTAIN COLUMN TYPES
df.printSchema()
df.dtypes # Does the same

# OBTAIN COLUMN NAMES / LIST COLUMN NAMES / LIST COLUMNS
df.schema.names
df.columns

# NUMBER OF ROWS OF A SPARK DATAFRAME
df.count()
# NUMBER OF COLUMNS OF A SPARK DATAFRAME
len(df.columns)

# VIEW DATAFRAMES
display(df)
# HEAD OF DATAFRAMES
df.show(4)

# ACCESS ELEMENT OF A SPARK DATAFRAME / ACCESSING CELLS
# Access row number 5 of mock variable: "Variable_1"
df.where(df.id == 5).select('Variable_1').collect()[0]['Variable_1'] 
# To access a single value of a Spark DataFrame you need to first create an index column, standardized with the name 'id':
df = df.withColumn("id", monotonicallyIncreasingId())

# ASSIGNING PARTICULAR VALUE TO A CELL DATAFRAME / ASSIGNATION
# We are supposing that our df has 3 variables, and we want to change the cell of variable_3:
df = df.withColumn(i, \ when(((df["variable_1"] == value_of_variable_1) & (df["variable_2"] == value_of_variable_2)) , (new_value_of_variable_3).otherwise(df["variable_3"]))

# MISSING VALUES / COUNT NUMBER OF MISSINGS
df.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df.columns)).show() # prints it trough console
df.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df.columns)) # saves it as a dataframe

# DATA TYPE CONVERSION
from pyspark.sql.types import IntegerType
df = df.withColumn("name_of_column", df["name_of_column"].cast(StringType()))
# Convert several columns to String:
to_str = ['variable_to_str_1','variable_to_str_2','variable_to_str_3', ...]
for col in to_str:
  df = df.withColumn(col, df[col].cast(StringType()))
# Same as the for expression but in one-liner:
perf = perf.select([col(c).cast(StringType()).alias(c) for c in perf_to_str])
  
# ACTIVATE MARKDOWN COMMANDS
%md
type_your_markdown_lines
[close_the_cell_you_are_in]

# EDIT MARKDOWN CELL
[Do double-click inside the cell]


