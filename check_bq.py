import os
from google.cloud import bigquery
from dotenv import load_dotenv

# 1. Load your .env file
load_dotenv()

def verify_bigquery_connection():
    print("🔍 [SYSTEM] Initiating BigQuery Authentication Check... for testing")
    
    # Verify if the environment variable points to your key
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    print(f"📂 [INFO] Looking for Key at: {creds_path}")

    if not creds_path or not os.path.exists(creds_path):
        print("❌ [ERROR] service-account.json not found in the root folder!")
        return

    try:
        # Initialize the BigQuery Client
        client = bigquery.Client()
        
        # Test 1: Project Connection
        project = client.project
        print(f"✅ [SUCCESS] Connected to Project: {project}")
        
        # Test 2: Dataset Verification
        dataset_id = os.getenv("GCP_DATASET_ID", "stock_raw_data")
        dataset_ref = client.dataset(dataset_id)
        
        # Attempt to fetch metadata for your specific dataset
        dataset = client.get_dataset(dataset_ref)
        print(f"✅ [SUCCESS] Verified Dataset: {dataset.dataset_id}")
        print(f"📍 [LOCATION] Region: {dataset.location}")
        
    except Exception as e:
        print(f"❌ [CRITICAL] Connection Failed: {e}")

if __name__ == "__main__":
    verify_bigquery_connection()