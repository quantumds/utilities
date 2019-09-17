// CREATING A SPARK SESSION:
import org .apache.spark.sql.SparkSession

// READING A CSV FILE WITHOUT HEADER
val spark = SparkSession.builder().getOrCreate() # Create the Spark variable
val df = spark.read.csv(route_of_the_file_with_termination) # Read with Spark a CSV file
// Real life example:
val df = spark.read.csv("/1_spark_for_data_analysis_in_scala/train.csv") # Read with Spark a CSV file

// READING A CSV FILE WITH HEADER
val spark = SparkSession.builder().getOrCreate() # Create the Spark variable
val df_with_header = spark.read.option("header", true).csv(route_of_file_with_termination)
// Real life example
val df_2 = spark.read.option("header", true).csv("/1_spark_for_data_analysis_in_scala/train.csv")

// OBTAIN WORKING DIRECTORY:
System.getProperty("user.dir")

// TYPES OF A DATAFRAME
df_2.dtypes

// READING A CSV FILE WITH HEADER AND inferSchema
val spark = SparkSession.builder().getOrCreate() # Create the Spark variable
val df_with_header = spark.read.option("header", true).csv(route_of_file_with_termination)
val df_with_header_and_inferring_schema = spark.read.option("header", true).option("inferSchema", true).csv(route_of_file_with_termination)
// Real life example
val df_3 = spark.read.option("header", true).option("inferSchema", true).csv("/1_spark_for_data_analysis_in_scala/train.csv")

// SELECT
df.select("column_1", "column_2")

// FILTER
df.filter(df("column_to_filter") > reference_value).select("column_1", "column_2")

// CHANGE WORKING DIRECTORIES
// Change working directories is not suggested in scala. It is not a good practice.
