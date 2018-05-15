# LOAD PARQUET FILES
df = spark.read.parquet(route_of_your_parquet_file)
# OBTAIN HEADER
df.show(n=4) # n = number of lines that you want to see in the header of the file
# OBTAIN COLUMN TYPES
df.printSchema()
