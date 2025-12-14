SUMMARIZE (FROM read_parquet('datasets/taxis_str.parquet')
    SELECT  
    	date_trunc('second',strptime(pickup,'%Y-%m-%d %H:%M:%S.%n')) AS pickup,
    	date_trunc('second',strptime(dropoff,'%Y-%m-%d %H:%M:%S.%n')) AS dropoff,
    	CAST(passengers AS INTEGER) AS passengers,
        CAST(distance AS DOUBLE) AS distance,
        CAST(fare AS DOUBLE) AS fare,
        CAST(tip AS DOUBLE) AS tip,
        CAST(tolls AS DOUBLE) AS tolls,
        CAST(total AS DOUBLE) AS total,
    	color,payment,COLUMNS('.*_zone|.*_borough'));

-- CASTING dates in Secondes
CREATE OR REPLACE MACRO cast_to_datetime(col_date) AS date_trunc('second',strptime(col_date,'%Y-%m-%d %H:%M:%S.%n')); 

-- WINDOW functions
-- (1) : feature selected to preprocess
-- (2) : count this feauture to be able to order by greater 
WITH 
wt_pickup_zone AS (
  	SELECT pickup_zone,
    COUNT(1) as count_pickup_zone, 
    ROW_NUMBER() OVER (ORDER BY 2 asc) AS pickup_zone_enc
    FROM read_parquet('datasets/taxis_str.parquet')
  	GROUP BY 1 ORDER BY 2 DESC
    --ROW_NUMBER() OVER (ORDER BY pickup_zone) AS pickup_zone_enc
  	--FROM (SELECT DISTINCT pickup_zone FROM read_parquet('datasets/taxis_str.parquet'))
), 
wt_dropoff_zone AS (
    SELECT dropoff_zone, 
    COUNT(1) as count_dropoff_zone, 
    ROW_NUMBER() OVER (ORDER BY 2 asc) AS dropoff_zone_enc
	FROM read_parquet('datasets/taxis_str.parquet')
	GROUP BY 1 ORDER BY 2 DESC
),
wt_pickup_borough AS (
	SELECT pickup_borough, 
    COUNT(1) as count_pickup_borough, 
    ROW_NUMBER() OVER (ORDER BY 2 asc) AS pickup_borough_enc
	FROM read_parquet('datasets/taxis_str.parquet')
	GROUP BY 1 ORDER BY 2 DESC
),
wt_dropoff_borough AS (
	SELECT dropoff_borough, 
    COUNT(1) as count_drop_borough, 
    ROW_NUMBER() OVER (ORDER BY 2 asc) AS dropoff_borough_enc
	FROM read_parquet('datasets/taxis_str.parquet')
	GROUP BY 1 ORDER BY 2 DESC
),
-- ONEHOT ENCODER for feature color and payment
onehot_color AS (
    PIVOT (FROM read_parquet('datasets/taxis_str.parquet'))
    ON color
    USING coalesce(max(color = color)::INT, 0) AS onehot
    GROUP BY color
),
onehot_payment AS (
    PIVOT (FROM read_parquet('datasets/taxis_str.parquet'))
    ON payment
    USING coalesce(max(payment = payment)::INT, 0) AS onehot
    GROUP BY payment
),
-- JOIN window tables
wt_encoded AS (
    SELECT *,
    FROM read_parquet('datasets/taxis_str.parquet')
    JOIN wt_pickup_zone USING (pickup_zone)
    JOIN wt_dropoff_zone USING (dropoff_zone)
    JOIN wt_pickup_borough USING (pickup_borough)
    JOIN wt_dropoff_borough USING (dropoff_borough)
    INNER JOIN onehot_color USING (color)
    INNER JOIN onehot_payment USING(payment)
),
-- CASTING and SELECT features
-- FEATURING ONEHOT and ORDINAL encoder
wt_drop_features AS (
    FROM wt_encoded
    SELECT 
    	cast_to_datetime(pickup) AS pickup,
    	cast_to_datetime(dropoff) AS dropoff,
        CAST(passengers AS INTEGER) AS passengers,
        CAST(distance AS DOUBLE) AS distance,
        CAST(fare AS DOUBLE) AS fare,
        CAST(tip AS DOUBLE) AS tip,
        CAST(tolls AS DOUBLE) AS tolls,
        CAST(total AS DOUBLE) AS total,
    	pickup_zone_enc,
    	dropoff_zone_enc,
    	pickup_borough_enc,
     	dropoff_borough_enc,
     	COLUMNS(c -> c LIKE '%_onehot')
)
-- EXCLUDE 1 Column for each ONEHOT encoder
FROM wt_drop_features
SELECT * EXCLUDE (green_onehot,cash_onehot)

-- TRAINSET 80% : splitting dataset with different ways

--USING SAMPLE 80 PERCENT (bernoulli)
USING SAMPLE 80% (system, 42)
--USING SAMPLE reservoir(80%)
--USING SAMPLE 80 PERCENT
