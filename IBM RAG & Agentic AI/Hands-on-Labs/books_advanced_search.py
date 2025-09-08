# Importing the necessary modules from the chromadb package:
# chromadb is used to interact with the Chroma DB database,
# embedding_functions is used to define the embedding model
import chromadb
from chromadb.utils import embedding_functions

# Define the embedding function using SentenceTransformers
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create a new instance of ChromaClient to interact with the Chroma DB
client = chromadb.Client()

# Define the name for the collection to be created or retrieved
collection_name = "book_collection"

# Define the main function to interact with the Chroma DB
def main():
    try:
        # Create a collection in the Chroma database with a specified name, 
        # distance metric, and embedding function. In this case, we are using 
        # cosine distance
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "A collection for storing book data"},
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": ef
            }
        )
        print(f"Collection created: {collection.name}")

        # List of book dictionaries with comprehensive details for advanced search
        books = [
            {
                "id": "book_1",
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "genre": "Classic",
                "year": 1925,
                "rating": 4.1,
                "pages": 180,
                "description": "A tragic tale of wealth, love, and the American Dream in the Jazz Age",
                "themes": "wealth, corruption, American Dream, social class",
                "setting": "New York, 1920s"
            },
            {
                "id": "book_2",
                "title": "To Kill a Mockingbird",
                "author": "Harper Lee",
                "genre": "Classic",
                "year": 1960,
                "rating": 4.3,
                "pages": 376,
                "description": "A powerful story of racial injustice and moral growth in the American South",
                "themes": "racism, justice, moral courage, childhood innocence",
                "setting": "Alabama, 1930s"
            },
            {
                "id": "book_3",
                "title": "1984",
                "author": "George Orwell",
                "genre": "Dystopian",
                "year": 1949,
                "rating": 4.4,
                "pages": 328,
                "description": "A chilling vision of totalitarian control and surveillance society",
                "themes": "totalitarianism, surveillance, freedom, truth",
                "setting": "Oceania, dystopian future"
            },
            {
                "id": "book_4",
                "title": "Harry Potter and the Philosopher's Stone",
                "author": "J.K. Rowling",
                "genre": "Fantasy",
                "year": 1997,
                "rating": 4.5,
                "pages": 223,
                "description": "A young wizard discovers his magical heritage and begins his education at Hogwarts",
                "themes": "friendship, courage, good vs evil, coming of age",
                "setting": "England, magical world"
            },
            {
                "id": "book_5",
                "title": "The Lord of the Rings",
                "author": "J.R.R. Tolkien",
                "genre": "Fantasy",
                "year": 1954,
                "rating": 4.5,
                "pages": 1216,
                "description": "An epic fantasy quest to destroy a powerful ring and save Middle-earth",
                "themes": "heroism, friendship, good vs evil, power corruption",
                "setting": "Middle-earth, fantasy realm"
            },
            {
                "id": "book_6",
                "title": "The Hitchhiker's Guide to the Galaxy",
                "author": "Douglas Adams",
                "genre": "Science Fiction",
                "year": 1979,
                "rating": 4.2,
                "pages": 224,
                "description": "A humorous space adventure following Arthur Dent across the galaxy",
                "themes": "absurdity, technology, existence, humor",
                "setting": "Space, various planets"
            },
            {
                "id": "book_7",
                "title": "Dune",
                "author": "Frank Herbert",
                "genre": "Science Fiction",
                "year": 1965,
                "rating": 4.3,
                "pages": 688,
                "description": "A complex tale of politics, religion, and ecology on a desert planet",
                "themes": "power, ecology, religion, politics",
                "setting": "Arrakis, distant future"
            },
            {
                "id": "book_8",
                "title": "The Hunger Games",
                "author": "Suzanne Collins",
                "genre": "Dystopian",
                "year": 2008,
                "rating": 4.2,
                "pages": 374,
                "description": "A teenage girl fights for survival in a brutal televised competition",
                "themes": "survival, oppression, sacrifice, rebellion",
                "setting": "Panem, dystopian future"
            },
        ]

        # Create comprehensive text documents for each book
        book_documents = []
        for book in books:
            document = f"{book['title']} by {book['author']}. {book['description']} "
            document += f"Themes: {book['themes']}. Setting: {book['setting']}. "
            document += f"Genre: {book['genre']} published in {book['year']}."
            book_documents.append(document)

        # Adding book data to the collection with comprehensive metadata
        collection.add(
            ids=[book["id"] for book in books],
            documents=book_documents,
            metadatas=[{
                "title": book["title"],
                "author": book["author"],
                "genre": book["genre"],
                "year": book["year"],
                "rating": book["rating"],
                "pages": book["pages"]
            } for book in books]
        )

        # Retrieve all the items (documents) stored in the collection
        all_items = collection.get()
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")

        # Function to perform advanced book search
        def perform_book_search(collection):
            print("=== Book Similarity Search ===")
            
            # Similarity search for magical adventures
            print("\n1. Finding magical fantasy adventures:")
            results = collection.query(
                query_texts=["magical fantasy adventure with friendship and courage"],
                n_results=3
            )
            for i, (doc_id, document, distance) in enumerate(zip(
                results['ids'][0], results['documents'][0], results['distances'][0]
            )):
                metadata = results['metadatas'][0][i]
                print(f"  {i+1}. {metadata['title']} by {metadata['author']} - Distance: {distance:.4f}")
            
            print("\n=== Metadata Filtering ===")
            
            # Filter by genre
            print("\n2. Finding Fantasy and Science Fiction books:")
            results = collection.get(
                where={"genre": {"$in": ["Fantasy", "Science Fiction"]}}
            )
            for i, doc_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                print(f"  - {metadata['title']}: {metadata['genre']} ({metadata['rating']}★)")
            
            # Filter by rating
            print("\n3. Finding highly-rated books (4.3+):")
            results = collection.get(
                where={"rating": {"$gte": 4.3}}
            )
            for i, doc_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                print(f"  - {metadata['title']}: {metadata['rating']}★")
            
            print("\n=== Combined Search ===")
            
            # Combined search: dystopian themes with high ratings
            print("\n4. Finding highly-rated dystopian books:")
            results = collection.query(
                query_texts=["dystopian society control oppression future"],
                n_results=3,
                where={"rating": {"$gte": 4.0}}
            )
            for i, (doc_id, document, distance) in enumerate(zip(
                results['ids'][0], results['documents'][0], results['distances'][0]
            )):
                metadata = results['metadatas'][0][i]
                print(f"  {i+1}. {metadata['title']} ({metadata['year']}) - {metadata['rating']}★")
                print(f"     Distance: {distance:.4f}")

        perform_book_search(collection)
    except Exception as error:
        print(f"Error: {error}")

if __name__ == "__main__":
    main()