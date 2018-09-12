/* INTERESTING TOPICS */
-- SQL will never show NULLs in its resullts of queries. You need to bear that in mind. It filters the NULLs out of the results.
-- SQL does not differentiate between uppercase and lower case, so you can use both indifferently.
-- Every time that we use the value '%' which means 'something' (numbers or letters) we need to use the operator 'LIKE'.
-- Every time that we use the operator LIKE, we need to use the value '%', which means 'something' (numbers or letters).
-- Values can be referred to without quotes WHERE age > 29; or with quotes (single or double): WHERE age > '29';
-- The good practice in SQL is that text is referred with quotes ''; and numbers without them.
-- You cannot select all columns from 2 tables without 'WHERE' clause, because SQL will give you the Cartesian product.
-- You only need to specify table_name.column_name in cases where the columns that you want so select have the same name in different tables.
-- All JOIN operations must be accompanied by ON. ALWAYS!!!

/* SQL SKETCH */
SELECT
FROM
JOIN
ON
WHERE
ORDER BY

/* COMMENTS IN SQL */
-- Comments can be done with 2 hyphens: "--" (Everything that is written after these 2 hyphens is a comment. 
-- But this methodology only works for 1 liners. After entering 'Enter. after the 2 hyphens the text is not considered 
-- comment by SQL).
/* Comments are also written putting the string: "/*" afterwards the comment, and then closing the comment with this string:
*/

/* SELECT INFORMATION FROM TABLE */
-- Select all columns from one table:
SELECT * 
FROM tablename; /*Select all columns from a table */
-- Select one column:
SELECT name_of_column 
FROM table;
-- Select several columns from table:
SELECT column_1, column_2 
FROM table_name;
-- Select several columns from table:
SELECT column_1, column_2 
FROM table_name;
-- Select all columns from one table:
SELECT *
FROM table_name;
-- Select all columns from 2 different tables:
SELECT *
FROM table1, table2
WHERE table1.id1 = table2.id2;

/* OPERATORS */
-- Equal to: =
-- Different from: !=
-- Or: OR
-- And: AND
-- Like: LIKE (to be used with % in the value)
SELECT *
FROM CAR
WHERE model LIKE 'F%';
-- Between: BETWEEN
SELECT column1, column2
FROM table_name
WHERE column3 BETWEEN value_of_column1 AND value_of_column2;

/* VALUES */ 
-- %
-- Means something (any collection of numbers and letters). The value '%' needs to be used with the operator LIKE.
SELECT *
FROM car
WHERE brand LIKE 'F%'; -- For example, 'Ford' would be result
-- _
-- Means 'any single value' (1 number or 1 letter, this means: 1 character). The value '_' needs to be used with the operator LIKE.
SELECT *
FROM car
WHERE brand LIKE 'Volk_wagen'; -- For example, 'Volkswagen'
-- NULL / NOT NULL (Missing values)
-- The value NULL or the value NOT NULL identifies if there are missings or there aren't missings.
-- The value NULL or NOT NULL will always be used with the operator IS.
SELECT *
FROM car
WHERE price IS NOT NULL;

/* WHERE */
SELECT *
FROM table_name
WHERE column_name = column_value;

/* CREATE A TABLE COPY FROM ANOTHER ONE */
SELECT *
    INTO db.new_table
  FROM db.old_table
  
/* ADD A COLUMN TO A TABLE */
ALTER TABLE table_name
  ADD column_name column_TYPE;

/* JOIN */
-- Select all columns from an internal join:
SELECT *
FROM
table1 JOIN table2
ON table1.id1 = table2.id2;
-- Real example:
SELECT *
FROM 
person JOIN CAR
ON person.id = car.owner_id;
-- Select several columns from different tables in an internal join:
SELECT director.name, movie.title
FROM
director JOIN movie
WHERE director.id = movie.director_id;

/* AS */
-- Select column from a table and change it to 'column_other':
SELECT 
    column AS column_other
FROM
    table;
# More difficult example. Implement AS ion a JOIN process:
SELECT
	name, title AS movie_title
FROM
	director JOIN movie
ON
	director.id = movie.director_id;

/* ORDER BY */
# Order by 'column_fromtablename':
SELECT *
FROM tablename
ORDER BY column_fromtablename;
# Real life example:
SELECT *
FROM employees
ORDER BY salary;















