-- CREATE DATABASE instacart;
-- USE instacart;

-- CREATE TABLE aisles(aisles_id INT, aisles VARCHAR(100));
-- CREATE TABLE departments(department_id INT, department VARCHAR(100));
-- CREATE TABLE order_products__prior(order_id INT, product_id INT, add_to_cart_order INT, recordered INT);
-- CREATE TABLE order_products__train(order_id INT, product_id INT, add_to_cart_order INT, recordered INT);
-- CREATE TABLE orders(order_id INT, user_id INT, eval_set VARCHAR(20), order_number INT, order_dow INT, order_hour_of_day INT, days_since_prior_order INT);
-- CREATE TABLE products(product_id INT, product_name VARCHAR(255), aisle_id INT,  department_id INT);
-- SHOW TABLES;

-- SET autocommit = 0;
-- SET foreign_key_checks = 0;
-- SET unique_checks = 0;

-- LOAD DATA LOCAL INFILE'C:/Users/Lenovo/Data Science/Ds_projects_Ai/Instacart/aisles.csv'
-- INTO TABLE aisles
-- FIELDS TERMINATED BY ','
-- ENCLOSED BY '"'
-- LINES TERMINATED BY '\r\n'
-- IGNORE 1 ROWS;

-- LOAD DATA LOCAL INFILE 'C:/Users/Lenovo/Data Science/Ds_projects_Ai/Instacart/departments.csv'
-- INTO TABLE departments
-- FIELDS TERMINATED BY ','
-- ENCLOSED BY '"'
-- LINES TERMINATED BY '\r\n'
-- IGNORE 1 ROWS;

-- LOAD DATA LOCAL INFILE 'C:/Users/Lenovo/Data Science/Ds_projects_Ai/Instacart/order_products__prior.csv'
-- INTO TABLE order_products__prior
-- FIELDS TERMINATED BY ','
-- ENCLOSED BY '"'
-- LINES TERMINATED BY '\r\n'
-- IGNORE 1 ROWS;

-- LOAD DATA LOCAL INFILE 'C:/Users/Lenovo/Data Science/Ds_projects_Ai/Instacart/order_products__train.csv'
-- INTO TABLE order_products__train
-- FIELDS TERMINATED BY ','
-- ENCLOSED BY '"'
-- LINES TERMINATED BY '\r\n'
-- IGNORE 1 ROWS;

-- LOAD DATA LOCAL INFILE 'C:/Users/Lenovo/Data Science/Ds_projects_Ai/Instacart/orders.csv'
-- INTO TABLE orders
-- FIELDS TERMINATED BY ','
-- OPTIONALLY ENCLOSED BY '"'
-- LINES TERMINATED BY '\n'
-- IGNORE 1 ROWS;

-- LOAD DATA LOCAL INFILE 'C:/Users/Lenovo/Data Science/Ds_projects_Ai/Instacart/products.csv'
-- INTO TABLE products
-- FIELDS TERMINATED BY ','
-- ENCLOSED BY '"'
-- LINES TERMINATED BY '\n'
-- IGNORE 1 ROWS;

-- SHOW TABLES;
-- checks table size
-- SELECT COUNT(*) FROM orders;
-- SELECT COUNT(*) FROM order_products__train;
-- SELECT COUNT(*) FROM products;

-- DESC orders;
-- DESC order_products__train;

-- checks orders table
-- SELECT COUNT(*) AS total_orders,
--        COUNT(DISTINCT user_id) AS total_users
-- FROM orders;

-- order_product_mapping
-- SELECT COUNT(*) AS total_rows,
--        COUNT(DISTINCT order_id) AS unique_orders,
--        COUNT(DISTINCT product_id) AS unique_products
-- FROM order_products__train;

-- Top 10 most ordered products
-- SELECT p.product_name,
--        COUNT(*) AS total_orders
-- FROM order_products__train opt
-- JOIN products p
-- ON opt.product_id = p.product_id
-- GROUP BY p.product_name
-- ORDER BY total_orders DESC
-- LIMIT 10;

-- Top 10 most reordered products

-- SHOW COLUMNS FROM order_products__train;

-- ALTER TABLE order_products__train
-- CHANGE COLUMN recordered reordered INT;

-- ALTER TABLE order_products__prior
-- CHANGE COLUMN recordered reordered INT;


-- SHOW COLUMNS FROM order_products__train;

-- SELECT p.product_name,
--        SUM(opt.reordered) AS reorder_count
-- FROM order_products__train opt
-- JOIN products p
-- ON opt.product_id = p.product_id
-- GROUP BY p.product_name
-- ORDER BY reorder_count DESC
-- LIMIT 10;

-- which order timeing to order high
-- SELECT order_hour_of_day,
--        COUNT(*) AS total_orders
-- FROM orders
-- GROUP BY order_hour_of_day
-- ORDER BY total_orders DESC;

-- weekday vs weekend behavior
-- SELECT order_dow,
--        COUNT(*) AS total_orders
-- FROM orders
-- GROUP BY order_dow
-- ORDER BY total_orders DESC;

-- reorder rate per product
-- SELECT p.product_name,
--        COUNT(*) AS total_orders,
--        SUM(opt.reordered) AS reorders,
--        ROUND(SUM(opt.reordered)/COUNT(*)*100, 2) AS reorder_rate
-- FROM order_products__train opt
-- JOIN products p ON opt.product_id = p.product_id
-- GROUP BY p.product_name
-- HAVING total_orders > 100
-- ORDER BY reorder_rate DESC
-- LIMIT 10;

-- User Behavior Analysis
-- SELECT user_id,
--        COUNT(order_id) AS total_orders,
--        AVG(days_since_prior_order) AS avg_gap
-- FROM orders
-- GROUP BY user_id
-- ORDER BY total_orders DESC
-- LIMIT 10;

-- Create Feature Tables
-- CREATE TABLE user_features AS
-- SELECT o.user_id,
--        COUNT(o.order_id) AS total_orders,
--        AVG(o.days_since_prior_order) AS avg_days_between_orders,
--        AVG(o.order_hour_of_day) AS avg_order_hour
-- FROM orders o
-- GROUP BY o.user_id;

-- ML-ready dataset (SQL)
-- CREATE TABLE ml_dataset AS
-- SELECT
--     opt.order_id,
--     o.user_id,
--     opt.product_id,
--     uf.total_orders,
--     uf.avg_days_between_orders,
--     uf.avg_order_hour,
--     opt.add_to_cart_order,
--     opt.reordered
-- FROM order_products__train opt
-- JOIN orders o
--     ON opt.order_id = o.order_id
-- JOIN user_features uf
--     ON o.user_id = uf.user_id;

-- USE instacart;
-- SHOW TABLES;

DROP TABLE IF EXISTS ml_dataset;

CREATE TABLE ml_dataset AS
SELECT
    opt.order_id,
    o.user_id,
    opt.product_id,
    uf.total_orders,
    uf.avg_days_between_orders,
    uf.avg_order_hour,
    opt.add_to_cart_order,
    opt.reordered
FROM order_products__train opt
JOIN orders o
    ON opt.order_id = o.order_id
JOIN user_features uf
    ON o.user_id = uf.user_id;


    
    


















