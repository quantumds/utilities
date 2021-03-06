/* INTERESTING TOPICS */
-- SQL will never show NULLs in its resullts of queries. You need to bear that in mind. It filters the NULLs out of the results.
-- SQL does not differentiate between uppercase and lower case, so you can use both indifferently.
-- Every time that we use the value '%' which means 'something' (numbers or letters) we need to use the operator 'LIKE'.
-- Every time that we use the operator LIKE, we need to use the value '%', which means 'something' (numbers or letters).
-- Values can be referred to without quotes WHERE age > 29; or with quotes (single or double): WHERE age > '29';
-- The good practice in SQL is that text is referred with quotes ''; and numbers without them.
-- You cannot select all columns from 2 tables without 'WHERE' clause, because SQL will give you the Cartesian product.
-- You only need to specify table_name.column_name in cases where the columns that you want so select have the same name in different tables.
-- SELECT determines the columns that will appear in the column that is retrieved.
-- GROUP BY must always be used with an aggregation function.
-- In GROUP BY queries each column in the SELECT part must either be used later for grouping or it must be used with one of the functions.


/* SQL SKETCH */
SELECT
FROM
JOIN
ON
WHERE
ORDER BY
GROUP BY
HAVING


/* COMMENTS IN SQL */
-- Comments can be done with 2 hyphens: "--" (Everything that is written after these 2 hyphens is a comment. 
-- But this methodology only works for 1 liners. After entering 'Enter. after the 2 hyphens the text is not considered 
-- comment by SQL).
/* Comments are also written putting the string: "/*" afterwards the comment, and then closing the comment with this string:
*/


/* SELECT */
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


/* SELECT DISTINCT */
-- Eliminates duplicates in the columns specified in the query:
SELECT DISTINCT department
FROM employees;
-- All unique combinations in 2 columns:
SELECT DISTINCT department, position
FROM employees;


/* FUNCTIONS/*
-- SELECT count()
-- Count number of rows / Count number of registries:
SELECT
count(*)
FROM
employees; 
-- SELECT count(*) counts the rows of the table, SELECT COUNT(variable) counts the registries non NULL in that variable.
-- Count number of different registries of/in a specific variable:	
SELECT count(DISTINCT variable)
FROM table_name;
-- SELECT max()
# Find the maximum salary of table employees:
SELECT
max(salary)
FROM employees;
-- SELECT min()
# Find the minimum salary of table employees:
SELECT
min(salary)
FROM employees;
/* SELECT sum()*/	
SELECT
sum(price)
FROM goods
WHERE goods.product_type = 'Basic';
	
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
-- SELECT avg()
SELECT 
avg(salary)
FROM employees;


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
-- NULL / NOT NULL (Missing values) / Missings
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
-- JOIN is equal to INNER JOIN
-- All JOIN operations must be accompanied by ON. ALWAYS!!!
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


/* LEFT JOIN / LEFT OUTER JOIN */
-- LEFT JOIN is all the registries from the table in the left plus the registries with same identifier or key in the right.
SELECT
  *
FROM car
LEFT JOIN person
  ON car.owner_id = person.id;
-- JOIN operations can also be done from one table against itself:
-- For example, who are the roomates of Jack Pearson?
SELECT *
FROM
student as t1 JOIN student as t2
ON t1.room_id = t2.room_id
WHERE t1.name = 'Jack Pearson'
and t1.id <> t2.id;

/* RIGHT JOIN / RIGHT OUTER JOIN */
-- RIGHT JOIN is all the registries from the table in the right plus the registries with same identifier or key in the left.
SELECT
  *
FROM car 
RIGHT JOIN person
  ON car.owner_id = person.id;


/* FULL JOIN */
-- FULL JOIN IS THE UNION OF LEFT JOIN and RIGHT JOIN, it is showing all the registries in the left table, in the right table, and add the ones common by the same key.
SELECT
  *
FROM car
FULL JOIN person
  ON car.owner_id = person.id;
  
  
/* NATURAL JOIN */
-- NATURAL JOIN is a type of JOIN where we do not use the ON line. This is because automatically the key are the columns with the same name.
SELECT
  * 
FROM person 
NATURAL JOIN car;

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
-- ORDER BY normally shows its result with ASC by default. This is in ascending way. 
-- Order by 'column_fromtablename':
SELECT *
FROM tablename
ORDER BY column_fromtablename;
-- Real life example:
SELECT *
FROM employees
ORDER BY salary;
-- ORDER BY with option ASC:
SELECT
  * 
FROM orders 
ORDER BY salary ASC;
-- ORDER BY with option DESC:
SELECT
  * 
FROM orders 
ORDER BY salary DESC;
-- ORDER BY several columns in ASC and DESC at the same time:
SELECT
  * 
FROM order 
ORDER BY salary ASC, age DESC;


/* GROUP BY */
SELECT
department,
min(salary),
max(salary)
FROM employees
WHERE year = 2018
GROUP BY (department);


/* HAVING */
-- HAVING is eclusively used to filter results inside GROUP BY´S:
SELECT
  customer_id, 
  order_date, 
  sum(total_sum) 
FROM orders 
GROUP BY customer_id, order_date 
HAVING sum(total_sum) > 2000;
-- Another case using WHERE and HAVING, where is about a column that is not included in the aggregation (salary):
SELECT department, avg(salary)
FROM employees
WHERE year =2012
GROUP BY department
HAVING ((avg(salary) > 3000));


/* ALIASES FOR TABLES */
-- Aliases for tables is used to right less. We use the particle AS after the tables. Same way as it is done with columns.
SELECT
  p.id, 
  p.name, 
  p.year, 
  c.id, 
  c.name, 
  c.year 
FROM person AS p 
JOIN car AS c 
  ON p.id = c.owner_id;

