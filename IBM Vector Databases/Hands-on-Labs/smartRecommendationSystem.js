const fs = require("fs");
const pdf = require("pdf-parse");
const { HfInference } = require("@huggingface/inference");
const readline = require("readline");
const { ChromaClient } = require("chromadb");

const hf = new HfInference("HUGGINGFACEHUB_API_TOKEN");
const chroma = new ChromaClient();
const collectionName = "job_collection";
const jobPostings = require('./jobPostings.js');

const extractTextFromPDF = async (filePath) => {
    try {
      const dataBuffer = fs.readFileSync(filePath);
      const data = await pdf(dataBuffer);
      const text = data.text.replace(/\n/g, " ").replace(/ +/g, " ");
      return text;
    } catch (err) {
      console.error("Error extracting text from PDF:", err);
      throw err;
    }
  };

const generateEmbeddings = async (text) => {
    try {
      const result = await hf.featureExtraction({
        model: "sentence-transformers/all-MiniLM-L6-v2",
        inputs: text,
      });

  return result
    } catch (err) {
      console.error("Error converting text to embeddings:", err);
      throw err;
    }
  };

const promptUserInput = (query) => {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });
    return new Promise((resolve) =>
        rl.question(query, (answer) => {
        rl.close();
        resolve(answer);
        })
    );
};

const storeEmbedding = async (jobPostings) => {
    const metadatas = jobPostings.map(() => ({}));

    const descriptions = jobPostings.map((job) => job.jobDescription.toLowerCase());

    const embeddings = await generateEmbeddings(descriptions);

    const ids = jobPostings.map((_, index) => index.toString());
    const documents = jobPostings.map(job => job.jobTitle);

    try {
        const collection = await chroma.getOrCreateCollection({ name: collectionName });
        
        await collection.add({
            ids,
            documents,
            embeddings,
            metadatas,
        });

        console.log("Stored embeddings in Chroma DB.");
    } catch (error) {
        console.error("Error storing embeddings in Chroma DB:", error);
        throw error;
    }
};

const main = async () => {

    try {
        await storeEmbedding(jobPostings);
      
        const filePath = await promptUserInput("Enter the path to the PDF: ");

        const text = await extractTextFromPDF(filePath);
        const embeding = await generateEmbeddings(text.toLowerCase());

        const collection = await chroma.getCollection({ name: collectionName });

        const results = await collection.query({
            queryEmbeddings: [embeding],
            n: 3,
        });

        if (results.ids.length > 0 && results.ids[0].length > 0) {
            const first3Items = results.ids[0].slice(0, 3);

            first3Items.forEach((id, index) => {
            const recommendedItem = jobPostings[parseInt(id)];

            console.log(`Top ${index + 1} Recommended ==> ${recommendedItem.jobTitle}`);
            });
        
        } else {
            console.log("No similar items found.");
        }

    } catch (err) {
        console.error("An error occurred:", err);
    }
};

main();