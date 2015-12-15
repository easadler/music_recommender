### DAY 3-6

I had a busier schedule at Galvanize and was spending a lot of time reviewing linear algebra. Dr. Mann recommended 'Linear Algebra Done Right' and I am very happy with the textbook. I am finally, getting to go through a proof based linear algebra course, which did not exist when I was a student at UW. 

After taking a subset of 10,000 random rows and looking through it, it quickly became apparent that the most interesting data exploration would need to be in the format required for collaborative filtering. This means that I would need every all the rows for from the table for each user in my subsample. Remeber this is the format of my data:

| userid | artist      | plays |
|--------|-------------|-------|
| user1  | radiohead   | 300   |
| user1  | the beatles | 100   | 
| user1  | kanye west  | 30    |   

At first I adapted my query from the previous blog post to random select 1000 userids. I then used the following query to get my data:
```SQL
WITH users as(
	SELECT userid FROM usa_plays 
		where row_id in 
			(SELECT round(random() * 21e6)::integer as id FROM generate_series(1, 1000)) 
		group by row_id 
		limit 10000
)
SELECT * FROM usa_plays WHERE userid IN users;
```
This turned out to be incredibly slow, easily over 3 minutes. I was told by my coworker a little trick which reduced query time from 3 minute plus to a handful of seconds.

```SQL
WITH users as(
	SELECT userid FROM usa_plays 
		where row_id in 
			(SELECT round(random() * 21e6)::integer as id FROM generate_series(1, 10000)) 
		group by row_id 
		limit 10000
)

SELECT temp.userid, usa_plays.artist, usa_plays.plays FROM usa_plays
	LEFT JOIN users
	on temp.userid = usa_plays.userid
	WHERE temp.userid IS NOT null;
```

Notice I left joined the users table and selected where the userid was not null, which grabs all the rows with userids in the random sample. A join and single binary comparison is much faster than scanning a list of 10000 userids once for each row of my giant table. 

Now it was finally time for some EDA. After some summary statistics, it became clear that there were noisey users and not all the artists were very popular. In order to find these different groups, I calculuted the mean, std, percentiles, count, kurtosis, and skew for both rows and columns. In this case, I thought that clustering on these statistics could seperate by group effectively for a few reasons:
1. Some users had way too many plays, 10,000's
2. Certain artists had many of plays but only a few different listers
3. Some users listed to many artists, but only a few listens for each artist
