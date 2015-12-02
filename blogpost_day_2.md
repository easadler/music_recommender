### DAY 2

I did not get to dedicate much time to this project today and unfortunately most of it was spent trouble shooting postgresql. Fortunately, I did learn some postgresql and command line tricks along with set myself up for easy EDA using an ipython notebook, pandas, and psycopg2. 

I am really impressed by the easy workflow from psycopg2 to pandas. The following code shows how simple it is to connect postgres to pandas. 

```python
import psycopg2 as pg2
import pandas as pd

# Connect to psql database
conn = pg2.connect(dbname='dbname', user='user', host='/tmp')
c = conn.cursor()

# Query database for random sample of 10000 rows
query = 'SELECT * FROM table'
df = pd.read_sql_query(query, conn)
```
The read_sql_query() method in pandas accepts a query as a string and the connection from psycop2. Running this code after changing names will painlessly transfer data from postgres into a pandas dataframe where some serious analysis can begin. 

The dataset I am using is just over 17.5 million rows, so loading the whole dataset in pandas would kill my macbook. I wanted to find a fast way to randomly sample the data. I found a [blog post](https://www.periscopedata.com/blog/how-to-sample-rows-in-sql-273x-faster.html) that gave a very fast solution. It is 273 times faster than the obvious ording by random() and taking the top results. Here is the optimized query:
```sql
SELECT * FROM users
where row_id IN (
  SELECT round(random() * 1.8e7)::integer AS id
  FROM generate_series(1, 10001)
  GROUP BY id
)
LIMIT 10000;
```

This query generates a random series of length 11,000 and then groups by id because there could be duplicates. With a 1/1.8e7 chance of even drawing a particular integer, I'm not too worried. Just in case, I will increase the series length by 1. Now I want to figure out the exact probability! 

This solution requires an ascending id column, so I had to add a new column to the table using the following command:
```sql
ALTER TABLE user_plays ADD COLUMN row_id SERIAL PRIMARY KEY;
```
After a few seconds, I had the necesarry column and I was able to do some quick EDA. I soon realized that my sampled dataset had a maximum row id of 8.6 million dollars. On a quick investigation, I discovered that my table was 10M rows short! 

After some exploration on the command line using sed, I discovered that there were some non-ascii characters, bad userid values, and double quotes dispersed throughout the 17M rows. After delete my table and then executing the following lines, I was up and running with 17M rows. 

I also learned that postgresql defaults for [work_mem](http://www.postgresql.org/docs/current/static/runtime-config-resource.html#GUC-WORK-MEM) and [maintenance_work_mem](http://www.postgresql.org/docs/current/static/runtime-config-resource.html#GUC-MAINTENANCE-WORK-MEM) are very low. Surprisingly less than 100 megabytes each. Every task that requires memory, i.e. sorting or altering a table, was limited to under 100 megabytes. After increasing the limits, up to 2 gigabytes for work_mem, my queries, table deletions, and alterations were fast even with a 17M row dataset. 

```unix
sed -i.bak '/sep 20, 2008/d' ratings.tsv 
perl -pi -e 's/[[:^ascii:]]//g' ratings.tsv
sed -i.bak 's/\"//g' ratings.tsv
```

I was only able to do some quick EDA and data cleaning, which I will expand on tomorrow if my DSR duties don't take too long. Outside of building a recommender, I could answer interesting questions about users of different genres (discovered through clustering). I am interested in the distribution of plays for users of different genres and discovering artists that transcend genre and those that link others together. Here is a [link]() to my ipython notebook from today.

Cheers!

