import psycopg2
 
DB_PARAMS = {
    "dbname": "hybrid_rag_db",
    "user": "rag_app_user",
    "password": "ragapp",
    "host": "localhost",
    "port": "5432"
}
 
def check_postgres_connection():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        conn.close()
        print("✅ PostgreSQL connection successful. Tables are ready.")
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
 
 
check_postgres_connection()