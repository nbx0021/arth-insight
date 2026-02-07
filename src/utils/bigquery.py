import os
from google.cloud import bigquery
from google.api_core import exceptions

def get_bq_client():
    """
    Returns a BigQuery Client using credentials from the .env path.
    """
    try:
        # This looks for the path you set in .env
        client = bigquery.Client()
        return client
    except exceptions.GoogleAuthError as e:
        print(f"❌ Auth Error: Check your service-account.json: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return None

def execute_query(query):
    """
    Executes a SQL query and returns a Pandas DataFrame.
    """
    client = get_bq_client()
    if client:
        try:
            query_job = client.query(query)
            return query_job.to_dataframe()
        except Exception as e:
            print(f"❌ Query Failed: {e}")
            return None
    return None