# get 10,000 random samples
SELECT * FROM users
where row_id IN (
  SELECT round(random() * 21e6)::integer AS id
  FROM generate_series(1, 11000)
  GROUP BY id
)
LIMIT 10000;



##### GET USA SUBSET ####

CREATE TABLE usa_plays AS SELECT up.userid as userid, up.artist as artist , up.plays as plays FROM users as u
	LEFT JOIN user_plays as up
	on u.userid = up.userid
	and u.country = 'United States';

ALTER TABLE usa_plays ADD COLUMN row_id SERIAL PRIMARY KEY;