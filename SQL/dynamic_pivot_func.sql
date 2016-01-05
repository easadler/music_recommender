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