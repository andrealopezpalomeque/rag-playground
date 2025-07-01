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
    for filename in glob.glob(f"{folder}/*.txt"):
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
            chunks = chunk_text(text, chunk_size)
            for idx, chunk in enumerate(chunks):
                docs.append(chunk)
                metadatas.append({'source': filename, 'chunk': idx})
    return docs, metadatas

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

print("Ask anything about your documents. Type 'quit' or just press enter to exit.\n")

# === INTERACTIVE LOOP STARTS HERE ===
def run_comparison():
  while True:
      # === 5. User enters their question === ------------------------------------------------------------------------------
      query = input("\nAsk your question: ").strip()
      if query.lower() in ("quit", "exit", ""):
          print("Goodbye!")
          break

      #TODO: usar la LLM para analizar la query, extraer sentencias con significado semantico y generar una query más precisa
      ## luego uso la query original mas las refinadas vas a tener mas embeding para luego correr la funcion de busqueda las veces que sea necesario

      # === 6. Embed the question as a vector ===
      query_embedding = embedder.encode([query]).tolist()

      # funcion de busqueda --------------------------------------------------------------------------------
      # === 7. Retrieve top-N most similar chunks to the question ===
      print("\n---[ CLASSIC QUERY EMBEDDING RAG ]---")
      results = collection.query(
          query_embeddings=query_embedding,
          n_results=10,
          include=["documents", "distances", "metadatas"]
      )

      # === 8. Filter results to only chunks containing query words (for transparency) ===
      filtered_docs = []
      query_words = [w.lower() for w in query.split() if len(w) > 2]  # skip very short words for relevance

      print("\nRetrieved Chunks (Query Embedding):")
      for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
          highlighted_doc = highlight_match(doc, query)
          print(f"\nChunk (distance: {dist:.4f}, source: {meta['source']}, chunk: {meta['chunk']}):")
          print(highlighted_doc)
          # Add to filtered_docs if any query word is in this chunk (case-insensitive)
          if any(word in doc.lower() for word in query_words):
              filtered_docs.append(doc)

      # Fallback: If no filtered docs, use the top 3 by similarity anyway
      if not filtered_docs:
          filtered_docs = results["documents"][0][:3]

      # === 9. Build context and prompt for the LLM ===
      context = "\n".join(filtered_docs)
      prompt = f"""You are a helpful assistant for answering questions about the book Alice in Wonderland.
  Only use the context below to answer, and quote the relevant sentence if possible.

  Context:
  {context}
  Question: {query}
  Answer:"""

      print("\nConstructed prompt for LLM:\n", prompt)

      # === 10. Call Ollama's LLM with the prompt and show the result ===
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
      if "message" in response_json and "content" in response_json["message"]:
          answer = response_json["message"]["content"]
      elif "content" in response_json:
          answer = response_json["content"]
      else:
          answer = str(response_json)

      print("\n[LLM Answer with Query Embedding]:", answer)

      # ----==[ HyDE: HYPOTHETICAL DOCUMENT EMBEDDING RAG ]==----
      print("\n\n---[ HyDE HYPOTHETICAL DOCUMENT EMBEDDING RAG ]---")
      hyde_prompt = f"Given the following question, write a plausible answer based only on your general knowledge (even if you need to guess):\n\nQuestion: {query}\nHypothetical Answer:"

      ollama_hyde_response = requests.post(
          "http://localhost:11434/api/chat",
          json={
              "model": "llama3",
              "messages": [
                  {"role": "user", "content": hyde_prompt}
              ],
              "stream": False
          }
      )
      hyde_json = ollama_hyde_response.json()
      if "message" in hyde_json and "content" in hyde_json["message"]:
          hypothetical_answer = hyde_json["message"]["content"]
      elif "content" in hyde_json:
          hypothetical_answer = hyde_json["content"]
      else:
          hypothetical_answer = query
      print("\n[HyDE Hypothetical Answer]:", hypothetical_answer)

      hyde_embedding = embedder.encode([hypothetical_answer]).tolist()

      hyde_results = collection.query(
          query_embeddings=hyde_embedding,
          n_results=10,
          include=["documents", "distances", "metadatas"]
      )

      hyde_filtered_docs = []
      hyde_words = [w.lower() for w in hypothetical_answer.split() if len(w) > 2]
      print("\nRetrieved Chunks (HyDE Embedding):")
      for doc, dist, meta in zip(hyde_results["documents"][0], hyde_results["distances"][0], hyde_results["metadatas"][0]):
          highlighted_doc = highlight_match(doc, hypothetical_answer)
          print(f"\nChunk (distance: {dist:.4f}, source: {meta['source']}, chunk: {meta['chunk']}):")
          print(highlighted_doc)
          if any(word in doc.lower() for word in hyde_words):
              hyde_filtered_docs.append(doc)
      if not hyde_filtered_docs:
          hyde_filtered_docs = hyde_results["documents"][0][:3]

      hyde_context = "\n".join(hyde_filtered_docs)
      hyde_final_prompt = f"""You are a helpful assistant for answering questions about the book Alice in Wonderland.
  Only use the context below to answer, and quote the relevant sentence if possible.

  Context:
  {hyde_context}
  Question: {query}
  Answer:"""

      print("\nConstructed prompt for LLM (HyDE):\n", hyde_final_prompt)

      hyde_llm_response = requests.post(
          "http://localhost:11434/api/chat",
          json={
              "model": "llama3",
              "messages": [
                  {"role": "user", "content": hyde_final_prompt}
              ],
              "stream": False
          }
      )
      hyde_response_json = hyde_llm_response.json()
      if "message" in hyde_response_json and "content" in hyde_response_json["message"]:
          hyde_answer = hyde_response_json["message"]["content"]
      elif "content" in hyde_response_json:
          hyde_answer = hyde_response_json["content"]
      else:
          hyde_answer = str(hyde_response_json)
      print("\n[LLM Answer with HyDE]:", hyde_answer)

      print("\n======== COMPARISON DONE! ========")

if __name__ == "__main__":
    run_comparison()