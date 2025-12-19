
import duckdb
import json
import os

# Connect to an in-memory database
con = duckdb.connect(database=":memory:")

# Install and load the JSON extension
con.execute("INSTALL json;")
con.execute("LOAD json;")

# The JSON file path
file_path = "data/metadata/citations.json"

# The query to extract failed downloads
query = f"""
SELECT
    citation.paper_id,
    citation.title
FROM (
    SELECT unnest(paper_entry.value.citations) AS citation
    FROM (
        SELECT unnest(map_entries(papers)) AS paper_entry
        FROM read_json('{file_path}', format='auto', maximum_object_size=999999999)
    )
)
WHERE
    citation.download_attempted = true AND citation.pdf_downloaded = false;
"""

try:
    result = con.execute(query).fetchall()

    if result:
        print("Articles that failed to download:")
        for row in result:
            print(f"- ID: {row[0]}, Title: {row[1]}")
    else:
        print("No failed PDF downloads found.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    con.close()
