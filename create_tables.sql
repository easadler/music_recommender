### Create table and load plays dataset


CREATE TABLE user_plays (
	userid character varying, 
	artistid character varying, 
	artist character varying, 
	plays int
);

\copy users FROM '/Users/evansadler/Desktop/recommender/lastfm-dataset-360K/usersha1-profile.tsv' with (format csv, delimiter E'\t');


### Create table and load user dataset

CREATE TABLE users (
	userid character varying, 
	gender char(1),
	age int,
	country character varying,
	signup date
);

\copy user_plays FROM '/Users/evansadler/Desktop/recommender/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv' with (format csv, delimiter E'\t');

### Queries for blog post

# Unique artists
SELECT COUNT(distinct artist)  FROM user_plays;

# Unique countries

SELECT COUNT(distinct country)  FROM users;

# both really slow, need to optimize thanks to sizable dataset