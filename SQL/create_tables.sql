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
