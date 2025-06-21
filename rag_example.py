import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests # Used for calling Ollama's local LLM API
import glob
import re

# === Function to highlight words from the query in the text for easier inspection ===
def highlight_match(text, query):
    # Highlight query words in yellow using ANSI escape codes
    words = re.findall(r'\w+', query.lower())
    highlighted = text
    for word in set(words):
        # \033[93m = bright yellow, \033[0m = reset
        highlighted = re.sub(f"(?i)({re.escape(word)})", r"\033[93m\1\033[0m", highlighted)
    return highlighted

# === Chunking: Split large documents into smaller pieces ===
def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap # overlap ensures context is not lost at chunk boundaries
    return [c for c in chunks if c.strip()]

# === Load all .txt docs from a folder, chunk, and add metadata ===
def load_and_chunk_docs_with_metadata(folder, chunk_size=500):
    docs = []
    metadatas = []
    import glob
    for filename in glob.glob(f"{folder}/*.txt"):
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
            chunks = chunk_text(text, chunk_size)
            for idx, chunk in enumerate(chunks):
                docs.append(chunk)
                metadatas.append({'source': filename, 'chunk': idx})
    return docs, metadatas


# === Query Analysis: Use LLM to refine and expand user queries ===
# This function sends the original user query to an LLM to generate multiple reformulated versions,
# which helps capture different semantic aspects of the question and improves retrieval
def analyze_query_with_llm(query):
    prompt = f"""Given the following user query, please:
1. Extract the key concepts, entities, and search terms
2. Generate 3-5 reformulated search queries that capture different aspects of the original question
3. Return the results as a JSON array of strings

User query: {query}

Return format should be concise search queries only, like:
["reformulated query 1", "reformulated query 2", "reformulated query 3"]
"""
    
    try:
        # Send the prompt to Ollama's API to process with the LLM
        llm_response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
        )
         # Extract the LLM's response text
        response_json = llm_response.json()
        if "message" in response_json and "content" in response_json["message"]:
            analysis = response_json["message"]["content"]
        else:
            analysis = str(response_json)
        
         # Parse the LLM's response - ideally it would be a JSON array of query variations
        try:
            import json
            import re
            
            # Try to find and extract a JSON array from the response using regex
            match = re.search(r'\[.*\]', analysis, re.DOTALL)
            if match:
                reformulated_queries = json.loads(match.group(0))
                return reformulated_queries
        except:
           # If JSON parsing fails, try to extract quoted strings as individual queries
            queries = re.findall(r'"([^"]*)"', analysis)
            if queries:
                return queries
            
        # Last resort: if all parsing attempts fail, return the original query
        return [query]
        
    except Exception as e:
        # Handle any errors in the LLM processing and fall back to the original query
        print(f"Error analyzing query: {e}")
        return [query]

# === Vector Search: Convert query to embedding and search the collection ===
# This function takes a text query, converts it to a vector embedding,
# and performs semantic similarity search in the vector database
def search_collection(collection, query, embedder, n_results=20):
    # Convert the text query to a vector embedding using the same model as document embeddings
    # This ensures query and documents are in the same vector space for comparison
    query_embedding = embedder.encode([query]).tolist()
    
    # Search the collection using the query embedding
    # Returns the most semantically similar document chunks based on vector similarity
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,                        # Number of results to return
        include=["documents", "distances", "metadatas"]  # Include the text, similarity scores, and metadata
    )
    return results

def main():
    # === 1. Load and chunk the docs, collect metadata (source file, chunk number) ===
    docs, metadatas = load_and_chunk_docs_with_metadata("mydocs", chunk_size=500)
    if not docs:
        docs = ["The cat sat on the mat."]
        metadatas = [{"source": "sample", "chunk": 0}]

    # === 2. Embedding step: Convert each chunk to a vector ===
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = embedder.encode(docs).tolist()

    # === 3. Create an in-memory vector DB with Chroma ===
    chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = chroma_client.create_collection("rag_demo")

    # === 4. Store each chunk (with its embedding & metadata) in the vector DB ===
    for idx, (doc, emb) in enumerate(zip(docs, doc_embeddings)):
        collection.add(
            documents=[doc],
            embeddings=[emb],
            ids=[str(idx)],
            metadatas=[metadatas[idx]]
        )

    # === 5. User enters their question ===
    while True:
        try:
            query = input("Ask your question (or type 'quit' to exit): ").strip()
            if query.lower() in ['quit', 'exit']:
                print("Exiting RAG loop.")
                break

            # === NEW: Analyze and expand the query using LLM ===
            print("\nAnalyzing your query to improve search results...")
            expanded_queries = analyze_query_with_llm(query)
            print(f"Generated query variations: {expanded_queries}")
            
            # Add original query if not already included
            if query not in expanded_queries:
                all_queries = [query] + expanded_queries
            else:
                all_queries = expanded_queries
                
            # === 6. Search with all query variations ===
            all_results = []
            all_docs = {}  # Use dict to avoid duplicates
            
            for q in all_queries:
                print(f"\nSearching with query variation: '{q}'")
                results = search_collection(collection, q, embedder, n_results=10)
                
                # Process results from this query
                for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
                    # Use ID based on metadata to avoid duplicates
                    doc_id = f"{meta['source']}_{meta['chunk']}"
                    if doc_id not in all_docs or dist < all_docs[doc_id]['distance']:
                        all_docs[doc_id] = {
                            'document': doc,
                            'distance': dist,
                            'metadata': meta
                        }
            
            # Convert back to list and sort by distance
            all_results = list(all_docs.values())
            all_results.sort(key=lambda x: x['distance'])
            
            # === 7. Display results and highlight matches ===
            query_words = []
            for q in all_queries:
                query_words.extend([w.lower() for w in q.split() if len(w) > 2])
            query_words = list(set(query_words))  # Remove duplicates
            
            filtered_docs = []
            print("\n=== Retrieved context chunks ===")
            for result in all_results[:20]:  # Show top 20 results
                doc = result['document']
                dist = result['distance']
                meta = result['metadata']
                
                highlighted_doc = highlight_match(doc, " ".join(query_words))
                print(f"\nChunk (distance: {dist:.4f}, source: {meta['source']}, chunk: {meta['chunk']}):")
                print(highlighted_doc)
                
                # Add to filtered_docs if any query word is in this chunk
                if any(word in doc.lower() for word in query_words):
                    filtered_docs.append(doc)
            
            # Fallback: If no filtered docs, use the top 3 by similarity
            if not filtered_docs:
                filtered_docs = [r['document'] for r in all_results[:3]]

            # === 8. Build context and prompt for the LLM ===
            context = "\n".join(filtered_docs)
            prompt = f"""You are a helpful assistant for answering questions about the book Alice in Wonderland.
Only use the context below to answer, and quote the relevant sentence if possible.

Context:
{context}
Question: {query}
Answer:"""

            print("\nConstructed prompt for LLM:\n", prompt)

            # === 9. Call Ollama's LLM with the prompt and show the result ===
            ollama_response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "llama3",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False
                }
            )

            response_json = ollama_response.json()
            print("Ollama raw response:", response_json)

            if "message" in response_json and "content" in response_json["message"]:
                  answer = response_json["message"]["content"]
            elif "content" in response_json:
                  answer = response_json["content"]
            else:
                  answer = str(response_json)

            print("\nLLM Answer:", answer)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received. Exiting.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()