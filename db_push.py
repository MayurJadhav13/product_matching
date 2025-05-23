from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get URI from environment
uri = os.getenv("MONGO_URI")

# Create MongoDB client
client = MongoClient(uri, tls=True)

# Load the CSV
df = pd.read_csv("output_with_features.csv")

# Select database and collection
db = client["jtp_task"]
collection = db["product"]

# Convert DataFrame to dictionary and insert
records = df.to_dict(orient="records")
collection.insert_many(records)

print("Inserted records into MongoDB Atlas.")
