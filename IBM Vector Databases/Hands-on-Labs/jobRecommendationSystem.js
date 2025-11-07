const { ChromaClient } = require("chromadb");
const client = new ChromaClient();
const { HfInference } = require("@huggingface/inference");

const hf = new HfInference("HUGGINGFACEHUB_API_TOKEN");
const jobPostings = require('./jobPostings.js');
const collectionName = "job_collection";

async function generateEmbeddings(texts) {
    const results = await hf.featureExtraction({
        model: "sentence-transformers/all-MiniLM-L6-v2",
        inputs: texts,
    });
    return results;
}

async function classifyText(text, labels) {
    const response = await hf.request({
        model: "facebook/bart-large-mnli",
        inputs: text,
        parameters: { candidate_labels: labels },
    });
    return response;
}

async function extractFilterCriteria (query) {
    const criteria = { location: null, jobTitle: null, company: null, jobType: null };
    const labels = ["location", "jobTitle", "company", "jobType"];
   
    const result = await classifyText(query,labels);

    const highestScoreLabel = result.labels[0];
    const score = result.scores[0];

    if(score > 0.5){
        criteria[`${highestScoreLabel}`] = result.sequence;
    }

    return criteria;
}

async function performSimilaritySearch(collection, queryTerm, filterCriteria) {
    try {
        const queryEmbedding = await generateEmbeddings([queryTerm]);

        const results = await collection.query({
            collection: collectionName,
            queryEmbeddings: queryEmbedding,
            n: 3,
        });

        if (!results || results.length === 0) {
            console.log(`No items found similar to "${queryTerm}"`);
            return [];
        }

        let topJobPostings = results.ids[0].map((id, index) => {
            const job = jobPostings.find(job => job.jobId.toString() === id);
            return {
                id,
                score: results.distances[0][index],
                ...job,
            };
        }).filter(Boolean);

        return topJobPostings.sort((a, b) => a.score - b.score);
    } catch (error) {
        console.error("Error during similarity search:", error);
        return [];
    }
}

async function main() {
    const query = "Creative Studio";

    try {
        const collection = await client.getOrCreateCollection({ name: collectionName });
        const uniqueIds = new Set();

        jobPostings.forEach((job, index) => {
            while (uniqueIds.has(job.jobId.toString())) {
                job.jobId = `${job.jobId}_${index}`;
            }
            uniqueIds.add(job.jobId.toString());
        });

        const jobTexts = jobPostings.map((job) => `jobTitle: ${job.jobTitle}. jobDescription: ${job.jobDescription}. jobType: ${job.jobType}. jobLocation: ${job.location}`);
        const embeddingsData = await generateEmbeddings(jobTexts);

        await collection.add({
            ids: jobPostings.map((job) => job.jobId.toString()),
            documents: jobTexts,
            embeddings: embeddingsData,
        });

        const filterCriteria = await extractFilterCriteria(query);
        console.log('Filter Criteria:', filterCriteria);

        const initialResults = await performSimilaritySearch(collection, query, filterCriteria);

        initialResults.slice(0, 3).forEach(({jobTitle, jobDescription, jobType, company}, index) => {
            console.log(`Top ${index + 1} jobTitle: ${jobTitle}, jobType: ${jobType}, jobDescription: ${jobDescription}, Company: ${company}`);
        });

    } catch (error) {
        console.error("Error:", error);
    }
}

main();