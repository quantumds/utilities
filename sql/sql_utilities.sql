# SELECT INFORMATION FROM TABLE:
# Select all columns:
select * from tablename; /*Select all columns from a table */
# Select one column:
select name_of_column from table;
# Select several columns from table:
select column_1, column_2 from table table_name;

# CREATE A TABLE COPY FROM ANOTHER ONE
SELECT *
    INTO db.new_table
  FROM db.old_table
  
# ADD A COLUMN TO A TABLE
ALTER TABLE table_name
  ADD column_name column_TYPE;
