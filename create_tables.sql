### Create table and load plays dataset


CREATE TABLE user_plays (
	userid character varying, 
	artistid character varying, 
	artist character varying, 
	plays int
);


\copy user_plays FROM '/Users/evansadler/Desktop/recommender/lastfm-dataset-360K/new_ratings.tsv' WITH (FORMAT CSV, DELIMITER E'\t', NULL '');
ALTER TABLE user_plays ADD COLUMN row_id SERIAL PRIMARY KEY;

### Create table and load user dataset

CREATE TABLE users (
	userid character varying, 
	gender char(1),
	age int,
	country character varying,
	signup date
);

\copy users FROM '/Users/evansadler/Desktop/recommender/lastfm-dataset-360K/usersha1-profile.tsv' WITH (FORMAT CSV, DELIMITER E'\t');


######## SAMPLING ###########

# get 10,000 random samples
SELECT * FROM users
where row_id IN (
  SELECT round(random() * 21e6)::integer AS id
  FROM generate_series(1, 11000)
  GROUP BY id
)
LIMIT 10000;

#get pivoted user table


WITH sample AS (
SELECT * FROM users_table 
	WHERE userid IN (...)
)


##### GET USA SUBSET ####



CREATE TABLE usa_plays AS SELECT up.userid as userid, up.artist as artist , up.plays as plays FROM users as u
	LEFT JOIN user_plays as up
	on u.userid = up.userid
	and u.country = 'United States';

ALTER TABLE usa_plays ADD COLUMN row_id SERIAL PRIMARY KEY;



SELECT 3633110




SELECT COUNT(distinct(artist)) FROM usa_plays;
112195

######## MISC ############

WITH temp as (
SELECT userid FROM usa_plays 
	GROUP BY plays
	WHERE SUM(plays) > 100 and
	COUNT(plays) > 10
)

######## pivot table from random subset ##############

CREATE TABLE temp as
SELECT userid FROM usa_plays 
	where row_id in 
		(SELECT round(random() * 3.634e6)::integer as id FROM generate_series(1, 1000)) 
	group by row_id 
	limit 10000;


CREATE TABLE sample as 
SELECT temp.userid, usa_plays.artist, usa_plays.plays FROM usa_plays
	LEFT JOIN temp
	on temp.userid = usa_plays.userid
	WHERE temp.userid IS NOT null;


SELECT string_agg(distinct artist || ' integer', ',') FROM sample group by artist ;



CREATE or REPLACE FUNCTION dynamic_pivot (tablename varchar, rowc varchar, colc varchar, valc varchar, celldatatype varchar) RETURNS varchar language plpgsql AS $$
DECLARE
	dynsql1 varchar;
	dynsql2 varchar;
	columnlist varchar;
BEGIN
-- 1. retrieve list of column names.
	DROP TABLE IF EXISTS results;
	dynsql1 = 'SELECT string_agg(distinct ''"''||'||colc||'||''" '||celldatatype||''','','' ORDER BY ''"''||'||colc||'||''" '||celldatatype||''') FROM '||tablename||';';
	EXECUTE dynsql1 INTO columnlist;
-- 2. set up the crosstab query
	dynsql2 = 'CREATE TEMP TABLE results as SELECT * FROM crosstab (
		''SELECT '||rowc||','||colc||','||valc||' FROM '||tablename||' ORDER BY 1,2''
		,''SELECT distinct '||colc||' FROM '||tablename||' ORDER BY 1'')
		as newtable ('||rowc||' varchar,'||columnlist||');';
EXECUTE dynsql2;
RETURN dynsql2;
END;
$$;


SELECT dynamic_pivot('sample','userid','artist','plays','integer');
SELECT * FROM results;



# reset psql
SELECT pg_reload_conf();

'''
command line stuff:

# line counts
wc -l usersha1-artmbid-artname-plays.tsv
wc -l usersha1-profile.tsv



# find matching lines causing problems
sed -n '/sep 20, 2008/p' usersha1-artmbid-artname-plays.tsv
sed '/sep 20, 2008/d' usersha1-artmbid-artname-plays.tsv > new_ratings.tsv

## adds line number

sed -i.bak '/sep 20, 2008/d' ratings.tsv 
perl -pi -e 's/[[:^ascii:]]//g' ratings.tsv
sed -i.bak 's/\"//g' ratings.tsv
'''


