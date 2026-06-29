import os
import json
import requests
from kafka import KafkaConsumer
from pymongo import MongoClient

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
MONGO_URL = os.getenv("MONGO_URL", "mongodb://mongodb:27017")
SEARCH_SERVICE_URL = os.getenv("SEARCH_SERVICE_URL", "http://search_service:8000")
DB_NAME = "ecommerce"

def main():
    print(f"Ingestion service starting...")
    print(f"Connecting to MongoDB at {MONGO_URL}...")
    try:
        mongo_client = MongoClient(MONGO_URL)
        db = mongo_client[DB_NAME]
        products_coll = db["products"]
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return

    print(f"Connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS}...")
    consumer = None
    # Add retry loop for Kafka connection on boot
    import time
    for attempt in range(1, 11):
        try:
            consumer = KafkaConsumer(
                "product-updates",
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                group_id="search-indexer",
                auto_offset_reset="earliest"
            )
            print("Successfully connected to Kafka.")
            break
        except Exception as e:
            print(f"Kafka connection attempt {attempt}/10 failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            
    if not consumer:
        print("Failed to connect to Kafka. Exiting Ingestion Service.")
        return

    print("Subscribed to 'product-updates' topic. Listening for messages...")
    
    for message in consumer:
        try:
            product = message.value
            if not product or not isinstance(product, dict):
                print("Skipping invalid or empty message.")
                continue
                
            action = product.pop("_action", "upsert")
            product_id = product.get("_id")
            
            if not product_id:
                print("Skipping message: missing '_id'")
                continue
                
            print(f"Received action '{action}' for product ID: {product_id}")
            
            if action == "upsert":
                # 1. Write/Update MongoDB
                products_coll.update_one(
                    {"_id": product_id}, 
                    {"$set": product}, 
                    upsert=True
                )
                print(f"MongoDB: Upserted product {product_id}")
                
                # 2. Notify Search Service to add to FAISS
                # Put '_action' back for search service or keep clean
                product["_id"] = product_id
                try:
                    res = requests.post(
                        f"{SEARCH_SERVICE_URL}/admin/upsert_product", 
                        json=product, 
                        timeout=5
                    )
                    if res.status_code == 200:
                        print(f"Search Service: Registered upsert for product {product_id}")
                    else:
                        print(f"Search Service: Failed to register upsert. Status: {res.status_code}")
                except Exception as ex:
                    print(f"Search Service unavailable: {ex}. Search service will catch up on reboot.")
                    
            elif action == "delete":
                # 1. Delete from MongoDB
                products_coll.delete_one({"_id": product_id})
                print(f"MongoDB: Deleted product {product_id}")
                
                # 2. Notify Search Service to delete from FAISS
                try:
                    res = requests.post(
                        f"{SEARCH_SERVICE_URL}/admin/delete_product", 
                        json={"_id": product_id}, 
                        timeout=5
                    )
                    if res.status_code == 200:
                        print(f"Search Service: Registered delete for product {product_id}")
                    else:
                        print(f"Search Service: Failed to register delete. Status: {res.status_code}")
                except Exception as ex:
                    print(f"Search Service unavailable: {ex}")
                    
        except Exception as e:
            print(f"Error processing message: {e}")
            continue

if __name__ == "__main__":
    main()
