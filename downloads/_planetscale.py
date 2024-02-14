import sys
from pathlib import Path

from dotenv import load_dotenv
import os
import MySQLdb
import csv

# Load environment variables from the .env file
load_dotenv()

TABLE_NAME = "downloads"
CSV_PATH = Path(__file__).parent / "daily.csv"

# Connect to the database
connection = MySQLdb.connect(
    host=os.getenv("DATABASE_HOST"),
    user=os.getenv("DATABASE_USERNAME"),
    passwd=os.getenv("DATABASE_PASSWORD"),
    db=os.getenv("DATABASE"),
    autocommit=True,
    ssl_mode="VERIFY_IDENTITY",
    # See https://planetscale.com/docs/concepts/secure-connections#ca-root-configuration
    # to determine the path to your operating systems certificate file.
    ssl={"ca": os.getenv("CERT_PATH")},
)

cursor = None
try:
    # Create a cursor to interact with the database
    cursor = connection.cursor()

    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    print(f"Tables in the database: {[t[0] for t in tables]}")

    # Read the CSV file
    with open(CSV_PATH) as f:
        reader: csv.DictReader = csv.DictReader(f)
        columns = reader.fieldnames
        entries = list(reader)

    query = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
date date NOT NULL,
{'\n    '.join([f'{col} int,' for col in columns if col != 'date'])}
PRIMARY KEY (DATE)
);
    """
    print(query)
    cursor.execute(query)

    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    print(f"Now tables in the database: {[t[0] for t in tables]}")

    if entries:
        query = f"""
INSERT INTO {TABLE_NAME} ({", ".join(columns)}) VALUES ({", ".join("%s" for c in columns)})
ON DUPLICATE KEY UPDATE 
{',\n'.join([f'{col} = VALUES({col})' for col in columns if col != 'date'])};
        """
        print(query)
        batch = [[v if v != "" else None for k, v in entry.items()] for entry in entries]
        cursor.executemany(query, batch)

    # Confirm number of rows and print the last 5 rows
    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    print("Number of rows in the downloads table:", cursor.fetchone()[0])

    cursor.execute(f"SELECT * FROM {TABLE_NAME} ORDER BY date DESC LIMIT 5")
    print("Last 5 rows in the downloads table:")
    for row in cursor.fetchall():
        print(row)

except MySQLdb.Error as e:
    print("MySQL Error:", e)

finally:
    print("Closing connection")
    if cursor is not None:
        cursor.close()
    connection.close()
    sys.exit(1)
