# LOAD PARQUET FILES
df = spark.read.parquet(route_of_your_parquet_file)
# OBTAIN COLUMN TYPES
df.printSchema()
