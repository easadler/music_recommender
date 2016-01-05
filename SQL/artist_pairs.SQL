WITH clean_id AS (
	SELECT userid FROM usa_plays
	GROUP BY userid
	Having MIN(plays) < 100 AND (MAX(plays) BETWEEN 10 AND 1000);
), cleaned AS (
	SELECT temp.userid, usa_plays.artist, usa_plays.plays FROM usa_plays
	LEFT JOIN clean_id
	on temp.userid = usa_plays.userid
	WHERE temp.userid IS NOT null;
)


CREATE TABLE pairs2 AS 
WITH clean_id AS (
	SELECT userid FROM usa_plays
	GROUP BY userid
	Having MIN(plays) < 100 AND (MAX(plays) BETWEEN 10 AND 2000)
), cleaned AS (
	SELECT clean_id.userid, usa_plays.artist, usa_plays.plays FROM usa_plays
	LEFT JOIN clean_id
	on clean_id.userid = usa_plays.userid
	WHERE clean_id.userid IS NOT null
), temp as (
	SELECT artists.artist, artists.artistid, cleaned.userid FROM cleaned
	INNER JOIN artists
	ON artists.artist = cleaned.artist
), temp2 as (
select a1.artist as artist1, a2.artist as artist2, a1.artistid, a2.artistid from temp as a1
left join temp as a2
on a1.userid = a2.userid
WHERE a1.artistid < a2.artistid
)

SELECT COUNT(*), artist1, artist2
FROM temp2 
GROUP BY artist1, artist2; 