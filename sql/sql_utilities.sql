-- SELECT INFORMATION FROM TABLE:
-- Select all columns:
SELECT * 
FROM tablename; /*Select all columns from a table */
-- Select one column:
SELECT name_of_column 
FROM table;
-- Select several columns from table:
SELECT column_1, column_2 
FROM table_name;

-- WHERE
SELECT *
FROM table_name
WHERE column_name = column_value;

-- CREATE A TABLE COPY FROM ANOTHER ONE
SELECT *
    INTO db.new_table
  FROM db.old_table
  
-- ADD A COLUMN TO A TABLE
ALTER TABLE table_name
  ADD column_name column_TYPE;
