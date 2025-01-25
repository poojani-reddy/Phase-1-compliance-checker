import psycopg2
from psycopg2 import sql

def test_postgres_connection():
    # Define the connection string
    connection_string = "postgresql://postgres:password@localhost:5433/postgres"

    try:
        # Establish the connection
        conn = psycopg2.connect(connection_string)
        print("Connection to the PostgreSQL database was successful!")

        # Create a cursor object
        cursor = conn.cursor()

        # Execute a test query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"PostgreSQL version: {version[0]}")

        # Close the cursor and connection
        cursor.close()
        conn.close()
        print("Connection closed.")
    except Exception as e:
        print("An error occurred while connecting to the database:")
        print(e)

if __name__ == "__main__":
    test_postgres_connection()
