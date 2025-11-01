const { ChromaClient } = require('chromadb'); // Imports the ChromaClient class from the chromadb library
const client = new ChromaClient();            // Creates a new ChromaClient instance

async function main() {
    try {  
        // Place your asynchronous code here
        const collection = await client.getOrCreateCollection({
            name: "my_basic_collection" // Specifies the name of the collection to retrieve or create
        });

        const texts = [
            "This is sample text 1.",  // First text item in the array
            "This is sample text 2.",  // Second text item in the array
            "This is sample text 3."   // Third text item in the array
        ];

        // Create an array of unique IDs for each text item in the 'texts' array
        // Each ID follows the format 'document_<index>', where <index> starts from 1
        const ids = texts.map((_, index) => `document_${index + 1}`);

        // Use the 'add' method to insert documents into the 'collection'
        // Each document has a unique ID from the 'ids' array and content from the 'texts' array
        await collection.add({ ids: ids, documents: texts });

        // Retrieve all items currently stored in the 'collection'
        const allItems = await collection.get();

        // Log the retrieved items to the console for inspection
        console.log(allItems);
    } catch (error) {
        console.error("Error:", error);  // Catches and logs any error that occurs in the try block
    }
}

// Run the main function
main();