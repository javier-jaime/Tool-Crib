// Import ChromaClient and DefaultEmbeddingFunction from chromadb package
const { ChromaClient, DefaultEmbeddingFunction } = require('chromadb');
// Instantiate a new ChromaClient and DefaultEmbeddingFunction
const client = new ChromaClient();
const default_emd = new DefaultEmbeddingFunction(); // Use the default text embedding function
const collectionName = "my_grocery_collection11"; // Set the collection name

// Main function to execute the logic
async function main() {
    try {
        // Create or retrieve the collection from Chroma DB with specified name and embedding function
        const collection = await client.getOrCreateCollection({
            name: collectionName,
            embeddings: default_emd
        });

        // Sample list of grocery items to store in the collection
        const texts = [
            'fresh red apples',
            'organic bananas',
            'ripe mangoes',
            'whole wheat bread',
            'farm-fresh eggs',
            'natural yogurt',
            'frozen vegetables',
            'grass-fed beef',
            'free-range chicken',
            'fresh salmon fillet',
            'aromatic coffee beans',
            'pure honey',
            'golden apple',
            'red fruit'
        ];

        // Generate unique document IDs for each text item
        const ids = texts.map((_, index) => `document_${index + 1}`);

        // Generate embeddings for the texts using the default embedding function
        const embeddingsData = await default_emd.generate(texts);

        // Add the texts and their embeddings to the collection
        await collection.add({ ids: ids, documents: texts, embeddings: embeddingsData });

        // Retrieve and log all items in the collection
        const allItems = await collection.get();
        console.log(allItems);

        // Perform similarity search on the collection
        await performSimilaritySearch(collection, allItems);

    } catch (error) {
        // Catch and log any errors that occur during the main function execution
        console.error("Error:", error);
    }
}

// Function to perform similarity search on the collection based on a query term
async function performSimilaritySearch(collection, allItems) {
    try {
        // Query term to search for similar items
        const queryTerm = "apple"; // Replace with your query term

        // Perform the query to retrieve top 3 similar documents
        const results = await collection.query({
            collection: collectionName,
            queryTexts: [queryTerm],
            n: 3 // Retrieve top 3 results
        });

        // Log the results of the query
        console.log(results);

        // Check if no results are found and log a message
        if (!results || results.length === 0) {
            console.log(`No documents found similar to "${queryTerm}"`);
            return; // Exit the function if no results are found
        }

        // Log the top 3 similar documents found
        console.log(`Top 3 similar documents to "${queryTerm}":`);
        for (let i = 0; i < 3; i++) {
            // Get the document ID and score for each result
            const id = results.ids[0][i]; // Get ID from 'ids' array
            const score = results.distances[0][i]; // Get score from 'distances' array

            // Retrieve the document text based on the ID
            const text = allItems.documents[allItems.ids.indexOf(id)];

            // If text is not available, log a placeholder message
            if (!text) {
                console.log(` - ID: ${id}, Text: 'Text not available', Score: ${score}`);
            } else {
                // Log the document ID, text, and similarity score
                console.log(` - ID: ${id}, Text: '${text}', Score: ${score}`);
            }
        }
    } catch (error) {
        // Catch and log any errors during the similarity search
        console.error('Error during similarity search:', error);
    }
}

// Call the main function to execute the process
main();