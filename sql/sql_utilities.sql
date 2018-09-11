-- INTERESTING TOPICS
-- Values can be referred to without quotes WHERE age > 29; or with quotes (single or double): WHERE age > '29';
-- The good practice in SQL is that text is referred with quotes ''; and numbers without them.

-- COMMENTS IN SQL
-- Comments can be done with 2 hyphens: "--" (Everything that is written after these 2 hyphens is a comment. 
-- But this methodology only works for 1 liners. After entering 'Enter. after the 2 hyphens the text is not considered 
-- comment by SQL).
/* Comments are also written putting the string: "/*" afterwards the comment, and then closing the comment with this string:
*/

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

/* OPERATORS */
-- Equal to: =
-- Different from: !=
-- Or: OR
-- And: AND
-- Between:
SELECT column1, column2
FROM table_name
WHERE column3 BETWEEN value_of_column1 AND value_of_column2;

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
