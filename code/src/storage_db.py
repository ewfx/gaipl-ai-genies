import os
import json
import chromadb
import load_data as ld


def store_data_in_chromadb(df, embedding_model, db_path="./chroma_db", collection_name="incidents_dataset"):
    """
    Store data in ChromaDB.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be stored.
    embedding_model (SentenceTransformer): The model used to generate embeddings.
    db_path (str): The path to the ChromaDB database.
    collection_name (str): The name of the collection in ChromaDB.

    Returns:
    Collection: The ChromaDB collection.
    """
    # Ensure the directory exists
    os.makedirs(db_path, exist_ok=True)

    # Initialize ChromaDB client with persist_directory setting
    chroma_client = chromadb.PersistentClient(path=db_path)  # Database will persist on disk

    # Create a new collection
    collection = chroma_client.get_or_create_collection(name=collection_name)
    # if collection.count() > 0:
    #     collection.delete()  # Clear existing data

    # Check if required columns exist in the DataFrame
    required_columns = ['Incident Number', 'Short Description', 'Description', 'Connectivity Information', 'Affected Upstream', 'Affected Downstream', 'Resolution Notes']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the DataFrame")

    # Insert records into ChromaDB
    if collection.count() < 1:
        for index, row in df.iterrows():
            incident_text = f"{row['Short Description']} {row['Description']} {row['Connectivity Information']} {row['Affected Upstream']} {row['Affected Downstream']} {row['Resolution Notes']}"
            embedding = embedding_model.encode(incident_text).tolist()
            document = json.dumps(row.to_dict())  # Convert the row to a JSON string
            collection.add(
                ids=[str(row["Incident Number"])],  # Unique ID for each record
                documents=[document],  # Store the entire row as a JSON string
                embeddings=[embedding]  # Store the embeddings
            )

    print("✅ Data stored successfully in ChromaDB!")
    return collection



