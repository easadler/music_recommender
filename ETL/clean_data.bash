# remove random date in random columns
sed -i.bak '/sep 20, 2008/d' ratings.tsv 

# remove none ascii columns (chinese characters among others)
perl -pi -e 's/[[:^ascii:]]//g' ratings.tsv

#remove double qoutes
sed -i.bak 's/\"//g' ratings.tsv