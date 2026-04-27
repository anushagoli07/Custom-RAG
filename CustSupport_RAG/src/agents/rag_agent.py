"""Agentic RAG implementation using LangChain and LangGraph."""
import logging
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from config.config import GOOGLE_API_KEY, LLM_MODEL
from src.vector_store.faiss_store import FAISSVectorStore
from src.validation.validator import ValidationModule

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the RAG agent."""
    messages: Annotated[List, add_messages]
    query: str
    k: int
    retrieved_context: List[Dict[str, Any]]
    answer: str
    validation_results: Dict[str, Any]


class RAGAgent:
    """Agentic RAG agent using LangGraph."""
    
    def __init__(self, vector_store: FAISSVectorStore, validation_module: Optional[ValidationModule] = None, default_k: int = 5):
        """
        Initialize the RAG agent.
        
        Args:
            vector_store: FAISS vector store instance
            validation_module: Optional validation module
            default_k: Default number of chunks to retrieve
        """
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.vector_store = vector_store
        self.validation_module = validation_module or ValidationModule()
        self.default_k = default_k
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        # Create the agent graph
        self.agent = self._create_agent()
    
    def sanitize(self, obj):
        if isinstance(obj, dict):
            return {k: self.sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.sanitize(v) for v in obj]
        if hasattr(obj, "item"):
            return obj.item()
        return obj
    
    def _create_agent(self) -> StateGraph:
        """Create the LangGraph agent."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("validate", self._validate_node)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        # Add edges
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "validate")
        workflow.add_edge("validate", END)
        
        return workflow.compile()
    
    def _retrieve_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant context from vector store using LangChain."""
        query = state.get("query", "")
        if not query and state.get("messages"):
            # Extract query from last message
            last_message = state["messages"][-1]
            if isinstance(last_message, HumanMessage):
                query = last_message.content
            else:
                query = str(last_message.content)
        
        state["query"] = query
        
        # Get k from state or use default
        k = state.get("k", self.default_k)
        
        # Retrieve context using vector store search (which uses LangChain internally)
        retrieved_chunks = self.vector_store.search(query, k=k)
        state["retrieved_context"] = retrieved_chunks
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: {query}")
        return state
    
    def _generate_node(self, state: AgentState) -> AgentState:
        """Generate answer using LLM."""
        query = state["query"]
        retrieved_chunks = state.get("retrieved_context", [])
        
        # Build context from retrieved chunks
        context = "\n\n".join([
            f"Chunk {i+1} (Similarity: {chunk.get('similarity_score', 0):.3f}):\n{chunk.get('content', '')}"
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful customer support assistant. Use the provided context to answer questions accurately and concisely.
            
If the context doesn't contain enough information to answer the question, say so clearly.
Cite specific chunks when possible to support your answer."""),
            ("human", """Context:
{context}

Question: {query}

Please provide a helpful answer based on the context above.""")
        ])
        
        # Generate answer
        messages = prompt.format_messages(context=context, query=query)
        response = self.llm.invoke(messages)
        
        answer = response.content if hasattr(response, 'content') else str(response)
        state["answer"] = answer
        
        logger.info(f"Generated answer for query: {query}")
        return state
    
    def _validate_node(self, state: AgentState) -> AgentState:
        """Validate the answer and context."""
        query = state["query"]
        retrieved_chunks = state.get("retrieved_context", [])
        answer = state.get("answer", "")
        
        # Perform validation
        validation_results = self.validation_module.validate_complete(
            query=query,
            retrieved_chunks=retrieved_chunks,
            answer=answer
        )
        
        state["validation_results"] = validation_results
        
        logger.info(f"Validation completed. Confidence: {validation_results.get('overall_confidence', 0):.3f}")
        return state
    
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Process a query through the RAG agent.
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            
        Returns:
            Complete response with answer, context, and validation
        """
        try:
            # Run the agent
            initial_state = {
                "messages": [HumanMessage(content=question)],
                "query": question,
                "k": k,
                "retrieved_context": [],
                "answer": "",
                "validation_results": {}
            }
            
            final_state = self.agent.invoke(initial_state)
            
            return {
                "query": question,
                "answer": final_state.get("answer", ""),
                "retrieved_chunks": final_state.get("retrieved_context", []),
                "validation": self.sanitize(final_state.get("validation_results", {})),
                "confidence_score": float(final_state.get("validation_results", {}).get("overall_confidence", 0.0))
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
