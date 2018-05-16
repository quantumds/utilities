# LOAD PARQUET FILES
df = spark.read.parquet(route_of_your_parquet_file)
# OBTAIN HEADER
df.show(n=4) # n = number of lines that you want to see in the header of the file

# OBTAIN COLUMN TYPES
df.printSchema()
df.dtypes # Does the same

# OBTAIN COLUMN NAMES
df.schema.names

# VIEW DATAFRAMES
display(df)

# DATA TYPE CONVERSION
from pyspark.sql.types import IntegerType
df = df.withColumn("name_of_column", df["name_of_column"].cast(StringType()))
# Convert several columns to String:
to_str = ['variable_to_str_1','variable_to_str_2','variable_to_str_3', ...]
for col in to_str:
  df = df.withColumn(col, df[col].cast(StringType()))
  
# ACTIVATE MARKDOWN COMMANDS
%md
type_your_markdown_lines
[close_the_cell_you_are_in]




