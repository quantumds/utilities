# CREATE NEW DATA FRAME
# One way to do it:
df = sc.parallelize( [ (1,'female',233), 
                      (None,'female',314),
                      (0,'female',81),
                      (1, None, 342), 
                      (1, 'male', 109)]).toDF().withColumnRenamed("_1","survived").withColumnRenamed("_2","sex").withColumnRenamed("_3","count")
# Other way to do it:
df = spark.createDataFrame(
    [
     (1, 1.87, 'new_york'), 
     (4, 2.76, 'la'), 
     (6, 3.3, 'boston'), 
     (8, 4.1, 'detroit'), 
     (2, 5.70, 'miami'), 
     (3, 6.320, 'atlanta'), 
     (1, 6.1, 'houston')
    ],
    ('variable_1', "variable_2", "variable_3")
)

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

# FILTER A DATAFRAME
df.filter(df.name_of_variable > value_of_variable).collect()
df.where(df.name_of_variable == value_of_variable).collect()

# ASSIGNING PARTICULAR VALUE TO A CELL DATAFRAME / ASSIGNATION
# We are supposing that our df has 3 variables, and we want to change the cell of variable_3:
df = df.withColumn("variable_3", \ when(((df["variable_1"] == value_of_variable_1) & (df["variable_2"] == value_of_variable_2)) , new_value_of_variable_3).otherwise(df["variable_3"]))
# Other form of writing it is putting a break line after the back slash:
df = df.withColumn("variable_3", \ 
                   when(((df["variable_1"] == value_of_variable_1) & (df["variable_2"] == value_of_variable_2)) , new_value_of_variable_3).otherwise(df["variable_3"]))

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

# SELECT ONLY STRING VARIABLES
string_variables = [item[0] for item in df.dtypes if item[1].startswith('string')]

# FREQUENCY TABLE OF DATA FRAME / NUMBER OF LEVELS OF CATEGORICAL VARIABLES
for i in string_variables:
  print(i)
  print(len(df.select(i).distinct().rdd.map(lambda r: r[0]).collect()))
  print('-------------------------------------------')
  print()

# SELECT COLUMNS IN PYSPARK / SELECT VARIABLES / SELECT DATA
name_of_df = df.selectExpr("var1 as alias1", "var2 as alias2", "var3 as alias3", "var4 as alias4", ... , "varn as aliasn" )

# DROP A COLUMN IN SPARK / ERASE A COLUMN  
df.drop('name_of_column').collect()



select al columns except one
it is a 2 liner code


samples_only = samples_only.dropDuplicates(['idcrsampling'])
promo.groupBy('EventText').count().show()
samples.stat.crosstab("producttype", "defect").show()
samples_def_f = samples.filter(samples.defect == 'false').collect()
promo_only = promo_only.selectExpr("sn as sn", "IDEquipment as IDEquipment", "startdate as startdate", "EventCode as EventCode", "eventtext as eventtext", "analogval as analogval", "sampletime as samplingtime", "__index_level_0__ as __index_level_0__", "id as id")
promo_sample_inner = promo_only.join(samples_only, on=['samplingtime', 'sn'], how='inner')

