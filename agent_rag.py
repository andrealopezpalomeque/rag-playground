import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import json
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class AgentAction:
    action_type: str
    parameters: Dict[str, Any]
    reasoning: str

class RAGAgent:
    def __init__(self, collection, embedder, llm_model="llama3"):
        self.collection = collection
        self.embedder = embedder
        self.llm_model = llm_model
        self.action_history = []
        
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Agent analyzes the query to determine the best retrieval strategy"""
        analysis_prompt = f"""
        Analyze this query and determine the best retrieval strategy:
        Query: "{query}"
        
        Consider:
        1. Is this a factual question that needs specific information?
        2. Is this a complex question requiring multiple retrieval steps?
        3. Does this need clarification or refinement?
        4. What search terms would be most effective?
        
        Respond in JSON format:
        {{
            "query_type": "factual|complex|clarification",
            "confidence": 0.8,
            "search_terms": ["term1", "term2"],
            "strategy": "direct|hyde|multi_step|clarification",
            "reasoning": "explanation"
        }}
        """
        
        response = self._call_llm(analysis_prompt)
        try:
            return json.loads(response)
        except:
            return {"strategy": "direct", "query_type": "factual", "confidence": 0.5}
    
    def generate_search_variants(self, query: str) -> List[str]:
        """Generate multiple search variants for better retrieval"""
        variant_prompt = f"""
        Generate 3-5 different ways to search for information about this query.
        Include synonyms, related terms, and different phrasings.
        
        Original query: "{query}"
        
        Return as a JSON list of strings:
        ["variant1", "variant2", "variant3"]
        """
        
        response = self._call_llm(variant_prompt)
        try:
            variants = json.loads(response)
            return variants if isinstance(variants, list) else [query]
        except:
            return [query]
    
    def multi_step_retrieval(self, query: str, analysis: Dict) -> List[str]:
        """Perform multi-step retrieval based on agent analysis"""
        all_docs = []
        
        if analysis["strategy"] == "multi_step":
            # Step 1: Get search variants
            variants = self.generate_search_variants(query)
            
            # Step 2: Search with each variant
            for variant in variants:
                embedding = self.embedder.encode([variant]).tolist()
                results = self.collection.query(
                    query_embeddings=embedding,
                    n_results=5,
                    include=["documents", "distances", "metadatas"]
                )
                all_docs.extend(results["documents"][0])
            
            # Step 3: Deduplicate and rank
            unique_docs = list(set(all_docs))
            return unique_docs[:10]
            
        elif analysis["strategy"] == "hyde":
            return self._hyde_retrieval(query)
        else:
            return self._direct_retrieval(query)
    
    def _direct_retrieval(self, query: str) -> List[str]:
        """Your existing direct retrieval method"""
        embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=10,
            include=["documents", "distances", "metadatas"]
        )
        return results["documents"][0]
    
    def _hyde_retrieval(self, query: str) -> List[str]:
        """Your existing HyDE retrieval method"""
        hyde_prompt = f"Write a plausible answer to: {query}"
        hypothetical_answer = self._call_llm(hyde_prompt)
        
        embedding = self.embedder.encode([hypothetical_answer]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=10,
            include=["documents", "distances", "metadatas"]
        )
        return results["documents"][0]
    
    def evaluate_retrieval_quality(self, query: str, retrieved_docs: List[str]) -> Dict:
        """Agent evaluates if retrieved documents are sufficient"""
        eval_prompt = f"""
        Evaluate if these retrieved documents can answer the query adequately:
        
        Query: "{query}"
        
        Documents:
        {chr(10).join(retrieved_docs[:3])}
        
        Respond in JSON:
        {{
            "sufficient": true/false,
            "confidence": 0.8,
            "missing_info": ["what's missing"],
            "next_action": "answer|refine_search|ask_clarification"
        }}
        """
        
        response = self._call_llm(eval_prompt)
        try:
            return json.loads(response)
        except:
            return {"sufficient": True, "next_action": "answer"}
    
    def process_query(self, query: str) -> str:
        """Main agent processing loop"""
        print(f"\n🤖 Agent processing query: {query}")
        
        # Step 1: Analyze query
        analysis = self.analyze_query(query)
        print(f"📊 Analysis: {analysis['strategy']} strategy, confidence: {analysis['confidence']}")
        
        # Step 2: Retrieve documents
        retrieved_docs = self.multi_step_retrieval(query, analysis)
        
        # Step 3: Evaluate retrieval quality
        evaluation = self.evaluate_retrieval_quality(query, retrieved_docs)
        print(f"🔍 Retrieval evaluation: {evaluation['next_action']}")
        
        # Step 4: Take action based on evaluation
        if evaluation["next_action"] == "refine_search" and evaluation["confidence"] < 0.7:
            # Try a different strategy
            print("🔄 Refining search strategy...")
            analysis["strategy"] = "hyde" if analysis["strategy"] == "direct" else "multi_step"
            retrieved_docs = self.multi_step_retrieval(query, analysis)
        
        # Step 5: Generate final answer
        context = "\n".join(retrieved_docs[:5])
        final_prompt = f"""
        Answer this question using only the provided context:
        
        Context: {context}
        Question: {query}
        
        If the context doesn't contain enough information, say so clearly.
        Answer:
        """
        
        answer = self._call_llm(final_prompt)
        
        # Log the action
        action = AgentAction(
            action_type="query_processing",
            parameters={"strategy": analysis["strategy"], "docs_retrieved": len(retrieved_docs)},
            reasoning=analysis.get("reasoning", "")
        )
        self.action_history.append(action)
        
        return answer
    
    def _call_llm(self, prompt: str) -> str:
        """Your existing LLM call method"""
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
        )
        
        response_json = response.json()
        if "message" in response_json and "content" in response_json["message"]:
            return response_json["message"]["content"]
        return str(response_json)

# Integration with your existing code
def create_agent_rag_system():
    # Use your existing setup
    from rag_comparison import docs, metadatas, embedder, collection
    
    # Create the agent
    agent = RAGAgent(collection, embedder)
    
    print("🤖 Agent RAG System Ready! Type 'quit' to exit.\n")
    
    while True:
        query = input("\nAsk your question: ").strip()
        if query.lower() in ("quit", "exit", ""):
            print("Goodbye!")
            break
        
        # Let the agent process the query
        answer = agent.process_query(query)
        print(f"\n✅ Agent Answer: {answer}")
        
        # Show agent's action history
        if len(agent.action_history) > 0:
            last_action = agent.action_history[-1]
            print(f"\n📝 Agent Action: {last_action.action_type} - {last_action.reasoning}")


def run_agent_tests():
    """Run specific test scenarios to evaluate agent behavior"""
    from rag_comparison import docs, metadatas, embedder, collection
    agent = RAGAgent(collection, embedder)
    
    test_queries = [
        # Factual queries (should use direct retrieval)
        "When was Alice in Wonderland published?",
        "Who wrote Alice in Wonderland?",
        
        # Complex queries (should use multi-step)
        "Compare Alice's character development with Victorian literary themes",
        "How do mathematical concepts influence the story structure?",
        
        # Queries needing HyDE (abstract/conceptual)
        "What psychological themes are explored in the story?",
        "How does the story reflect Victorian society's view of childhood?",
        
        # Queries that might need refinement
        "What's the deal with the cat?",
        "Tell me about the tea party scene"
    ]
    
    print("🧪 Running Agent Test Suite\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {query}")
        print('='*60)
        
        answer = agent.process_query(query)
        print(f"\n✅ Answer: {answer}")
        
        if agent.action_history:
            last_action = agent.action_history[-1]
            print(f"📝 Strategy Used: {last_action.parameters.get('strategy', 'unknown')}")
            print(f"📊 Documents Retrieved: {last_action.parameters.get('docs_retrieved', 0)}")
        
        input("\nPress Enter to continue to next test...")

if __name__ == "__main__":
    # Add choice between interactive and test mode
    choice = input("Choose mode: (1) Interactive Agent (2) Test Suite: ").strip()
    if choice == "2":
        run_agent_tests()
    else:
        create_agent_rag_system()