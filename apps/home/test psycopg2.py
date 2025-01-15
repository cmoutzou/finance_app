import psycopg2

try:
    connection = psycopg2.connect(
        host="localhost",
        user="localhost",  # Replace with the new user
        password="root",  # Replace with the new password
        database="postgres"  # Connect to the default 'postgres' database
    )
    connection.autocommit = True
    cursor = connection.cursor()

    # Create a new database
    try:
        cursor.execute("CREATE DATABASE test_db;")
        print("Database created successfully")
    except psycopg2.errors.DuplicateDatabase:
        print("Database 'test_db' already exists. Continuing...")
        try:
            connection = psycopg2.connect(
                host='localhost',
                user='localhost',
                password='root',
                database='test_db'
            )
            connection.autocommit= True
            cursor=connection.cursor()

            create_table_query = '''
            CREATE TABLE IF NOT EXISTS employees (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                age INT,
                department VARCHAR(50)
            );
            '''
            cursor.execute(create_table_query)
            connection.commit()
            print("Table created successfully")

            # Insert data into the table
            insert_data_query = '''
            INSERT INTO employees (name, age, department)
            VALUES (%s, %s, %s);
            '''
            cursor.execute(insert_data_query, ("John Doe", 30, "Finance"))
            cursor.execute(insert_data_query, ("Jane Smith", 25, "IT"))
            connection.commit()
            print("Data inserted successfully")

            # Query the data
            select_query = "SELECT * FROM employees;"
            cursor.execute(select_query)
            rows = cursor.fetchall()

            # Print the query results
            for row in rows:
                print(f"ID: {row[0]}, Name: {row[1]}, Age: {row[2]}, Department: {row[3]}")
        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL database", error)
    except (Exception, psycopg2.Error) as error:
        print("Error while creating PostgreSQL database", error)
except (Exception, psycopg2.Error) as error:
    print("Error while creating PostgreSQL database", error)
finally:
    if 'cursor' in locals():
        cursor.close()
    if 'connection' in locals():
        connection.close()
    print("PostgreSQL connection is closed")
