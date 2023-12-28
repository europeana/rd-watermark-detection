import os

def get_file_metadata_in_batches(root_directory, batch_size=1000):
    batch = []
    count = 0
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            path = os.path.join(root, file)
            try:
                stat = os.stat(path)
                last_modified = datetime.fromtimestamp(stat.st_mtime)
                size = stat.st_size
                batch.append((path, last_modified, size))
                count += 1

                if count >= batch_size:
                    yield batch
                    batch = []
                    count = 0
            except FileNotFoundError:
                # File might have been deleted/modified during the process
                continue

    # Yield any remaining files in the last batch
    if batch:
        yield batch


import psycopg2

def batch_insert(db_connection, data):
    cursor = db_connection.cursor()
    records_list_template = ','.join(['%s'] * len(data))
    insert_query = f"INSERT INTO file_paths (path, last_modified, size) VALUES {records_list_template}"
    cursor.execute(insert_query, data)
    db_connection.commit()
    cursor.close()


# Database connection details
db_connection = psycopg2.connect(
    dbname="your_db", user="your_user", password="your_password", host="localhost", port ='5056'
)

root_directory = '/code/data'

for file_metadata_batch in get_file_metadata_in_batches(root_directory):
    batch_insert(db_connection, file_metadata_batch)

db_connection.close()
