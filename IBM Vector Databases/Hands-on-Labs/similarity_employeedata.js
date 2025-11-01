// Import required modules from Chroma
const { ChromaClient, DefaultEmbeddingFunction } = require('chromadb');

// Initialize the Chroma client and default embedding function
const client = new ChromaClient();
const collectionName = "employee_collection";
const default_emd = new DefaultEmbeddingFunction();

// Function to perform similarity search within the collection
async function performSimilaritySearch(collection, allItems) {
  try {
    // Define the query term (e.g., employee experience value) for similarity search
    const queryTerm = "3"; // Example query term (experience value as a string)

    // Generate embedding for the query term using the embedding function
    const queryEmbedding = await default_emd.generate([queryTerm]);

    // Perform similarity search in the collection using the query embedding
    const results = await collection.query({
      collection: collectionName,  // Specify the collection name to query against
      queryTexts: [queryTerm],     // Provide the query term (e.g., employee experience) in an array
      n: 3,                        // Specify the number of similar results to return
    });

    // Check if the results are empty or undefined
    if (!results || results.length === 0) {
      // Log a message if no similar documents are found for the query term
      console.log(`No documents found similar to "${queryTerm}"`);
      return;  // Exit the function if no similar results are found
    }

    // Log the header for the top 3 similar documents based on the query term
    console.log(`Top 3 similar documents to "${queryTerm}":`);

    // Loop through the top 3 results and log the document details
    for (let i = 0; i < 3; i++) {
      // Extract the document ID and similarity score from the results
      const id = results.ids[0][i];
      const score = results.distances[0][i];

      // Retrieve the document text corresponding to the current ID from allItems
      const text = allItems.documents[allItems.ids.indexOf(id)];

      // Check if the text is available; if not, log 'Text not available'
      if (!text) {
        console.log(` - ID: ${id}, Text: 'Text not available', Score: ${score}`);
      } else {
        // Log the document ID, text, and similarity score
        console.log(` - ID: ${id}, Text: '${text}', Score: ${score}`);
      }
    }
  } catch (error) {
    // Handle any errors that occur during the similarity search process
    console.error("Error during similarity search:", error);
  }
}

async function main() {
  try {
    // Create or retrieve the collection from the Chroma client
    const collection = await client.getOrCreateCollection({
      name: collectionName,  // Specify the collection name
      embeddings: default_emd // Set the default embedding function for the collection
    });

    // Example employee data to be added to the collection
    const employees = [
      { id: "employee_1", name: "John Doe", experience: 5, department: "Engineering", role: "Software Engineer" },
      { id: "employee_2", name: "Jane Smith", experience: 8, department: "Marketing", role: "Marketing Manager" },
      { id: "employee_3", name: "Alice Johnson", experience: 3, department: "HR", role: "HR Coordinator" },
      { id: "employee_4", name: "Michael Brown", experience: 12, department: "Engineering", role: "Senior Software Engineer" },
      { id: "employee_5", name: "Emily Wilson", experience: 2, department: "Marketing", role: "Marketing Assistant" },
      { id: "employee_6", name: "David Lee", experience: 15, department: "Engineering", role: "Engineering Manager" },
      { id: "employee_7", name: "Sarah Clark", experience: 8, department: "HR", role: "HR Assistant" },
      { id: "employee_8", name: "Chris Evans", experience: 20, department: "Engineering", role: "Senior Architect" },
      { id: "employee_9", name: "Jessica Taylor", experience: 4, department: "Marketing", role: "Marketing Specialist" },
      { id: "employee_10", name: "Alex Rodriguez", experience: 18, department: "Engineering", role: "Lead Software Engineer" },
      { id: "employee_11", name: "Hannah White", experience: 6, department: "HR", role: "HR Manager" },
      { id: "employee_12", name: "Kevin Martinez", experience: 2, department: "Engineering", role: "Chief Technology Officer" },
      { id: "employee_13", name: "Rachel Brown", experience: 7, department: "Marketing", role: "Marketing Director" },
      { id: "employee_14", name: "Matthew Garcia", experience: 3, department: "Engineering", role: "Junior Software Engineer" },
      { id: "employee_15", name: "Olivia Moore", experience: 12, department: "Engineering", role: "Principal Engineer" },
    ];

    // Extract employee experience values and convert them to strings
    const employeeExperiences = employees.map((employee) => employee.experience.toString());

    // Generate embeddings for employee experiences
    const embeddingsData = await default_emd.generate(employeeExperiences);

    // Add employee data and embeddings to the collection
    await collection.add({
      ids: employees.map((employee) => employee.id),
      employeeNames: employees.map((employee) => employee.name),
      documents: employees.map((employee) => employee.experience.toString()),
      embeddings: embeddingsData,
    });

    // Retrieve all items from the collection
    const allItems = await collection.get();

    // Perform similarity search based on the collection and allItems
    await performSimilaritySearch(collection, allItems);

  } catch (error) {
    // Handle any errors that occur in the main function
    console.error("Error in main function:", error);
  }
}

// Run the main function to initialize and perform the operations
main();
