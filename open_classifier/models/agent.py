"""Advanced classification agent with reasoning and tool integration."""

from typing import List, Dict, Any, Optional, Callable
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool, BaseTool
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage
import asyncio
import json
import time

from ..core.config import settings
from ..core.logging import struct_logger
from ..core.exceptions import ClassificationError, ModelLoadError
from .dspy_classifier import ClassifierModule
from .langchain_classifier import LangChainClassifier
from .embeddings import EmbeddingModel

class ClassificationTool(BaseTool):
    """Tool for text classification using multiple models."""
    
    name = "text_classifier"
    description = "Classify text into predefined categories using advanced ML models"
    
    def __init__(self, dspy_classifier: ClassifierModule, langchain_classifier: LangChainClassifier):
        super().__init__()
        self.dspy_classifier = dspy_classifier
        self.langchain_classifier = langchain_classifier
    
    def _run(self, text: str, use_ensemble: bool = True) -> str:
        """Run the classification tool."""
        try:
            if use_ensemble:
                # Get predictions from both models
                dspy_result = self.dspy_classifier.forward(text)
                langchain_result = asyncio.run(self.langchain_classifier.classify(text))
                
                # Ensemble the results
                if dspy_result["class"] == langchain_result["class"]:
                    confidence = (dspy_result["confidence"] + langchain_result["confidence"]) / 2
                    explanation = f"Both models agreed on '{dspy_result['class']}'. DSPy: {dspy_result['explanation']}. LangChain: {langchain_result['explanation']}"
                else:
                    # Use the more confident prediction
                    if dspy_result["confidence"] > langchain_result["confidence"]:
                        result = dspy_result
                        explanation = f"DSPy model more confident. {result['explanation']}"
                    else:
                        result = langchain_result
                        explanation = f"LangChain model more confident. {result['explanation']}"
                    confidence = max(dspy_result["confidence"], langchain_result["confidence"]) * 0.8
                
                return json.dumps({
                    "classification": dspy_result["class"] if use_ensemble else result["class"],
                    "confidence": confidence,
                    "explanation": explanation,
                    "ensemble_used": True
                })
            else:
                # Use only DSPy classifier
                result = self.dspy_classifier.forward(text)
                return json.dumps(result)
                
        except Exception as e:
            return json.dumps({
                "error": f"Classification failed: {str(e)}",
                "classification": "unknown",
                "confidence": 0.0
            })
    
    async def _arun(self, text: str, use_ensemble: bool = True) -> str:
        """Async version of the classification tool."""
        return await asyncio.to_thread(self._run, text, use_ensemble)

class SimilarityTool(BaseTool):
    """Tool for finding similar texts using embeddings."""
    
    name = "similarity_finder"
    description = "Find similar texts using semantic embeddings"
    
    def __init__(self, embedding_model: EmbeddingModel):
        super().__init__()
        self.embedding_model = embedding_model
    
    def _run(self, text: str, reference_texts: List[str], top_k: int = 3) -> str:
        """Find most similar texts."""
        try:
            similarities = self.embedding_model.find_similar(text, reference_texts, top_k)
            return json.dumps({
                "most_similar": similarities,
                "success": True
            })
        except Exception as e:
            return json.dumps({
                "error": f"Similarity search failed: {str(e)}",
                "success": False
            })
    
    async def _arun(self, text: str, reference_texts: List[str], top_k: int = 3) -> str:
        """Async version of similarity search."""
        return await asyncio.to_thread(self._run, text, reference_texts, top_k)

class ContextAnalysisTool(BaseTool):
    """Tool for analyzing text context and extracting insights."""
    
    name = "context_analyzer"
    description = "Analyze text context, sentiment, and extract key insights"
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__()
        self.llm = llm
    
    def _run(self, text: str) -> str:
        """Analyze text context."""
        try:
            prompt = f"""
            Analyze the following text and provide insights:
            
            Text: {text}
            
            Please provide:
            1. Overall sentiment (positive/negative/neutral)
            2. Key themes and topics
            3. Emotional tone
            4. Intent or purpose
            5. Target audience (if identifiable)
            
            Respond in JSON format.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return json.dumps({
                "error": f"Context analysis failed: {str(e)}",
                "analysis": "Could not analyze context"
            })
    
    async def _arun(self, text: str) -> str:
        """Async version of context analysis."""
        return await asyncio.to_thread(self._run, text)

class ClassificationAgent:
    """Advanced classification agent with reasoning and tool integration."""
    
    def __init__(self, labels: List[str] = None):
        self.labels = labels or settings.CLASS_LABELS
        self.logger = struct_logger.bind(component="classification_agent")
        
        if not settings.AGENT_ENABLED:
            raise ValueError("Agent functionality is disabled in settings")
        
        try:
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                openai_api_key=settings.OPENAI_API_KEY
            )
            
            # Initialize classifiers
            self.dspy_classifier = ClassifierModule(labels=self.labels)
            self.langchain_classifier = LangChainClassifier(labels=self.labels)
            self.embedding_model = EmbeddingModel()
            
            # Initialize tools
            self.tools = [
                ClassificationTool(self.dspy_classifier, self.langchain_classifier),
                SimilarityTool(self.embedding_model),
                ContextAnalysisTool(self.llm)
            ]
            
            # Create agent prompt
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_system_prompt()),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ])
            
            # Initialize memory
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=settings.AGENT_MEMORY_SIZE,
                return_messages=True
            )
            
            # Create agent
            self.agent = create_openai_functions_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=self.prompt
            )
            
            # Create executor
            self.executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                memory=self.memory,
                verbose=settings.DSPY_VERBOSE,
                max_iterations=settings.AGENT_MAX_ITERATIONS,
                handle_parsing_errors=True
            )
            
            self.logger.info("Classification agent initialized", labels=self.labels)
            
        except Exception as e:
            self.logger.error("Failed to initialize classification agent", error=str(e))
            raise ModelLoadError(f"Failed to initialize classification agent: {str(e)}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return f"""
        You are an advanced text classification agent with access to multiple AI models and analysis tools.
        
        Your primary task is to classify text into one of these categories: {', '.join(self.labels)}
        
        Available tools:
        1. text_classifier - Use this to classify text using ensemble models
        2. similarity_finder - Use this to find similar texts for context
        3. context_analyzer - Use this to analyze text context and sentiment
        
        Guidelines:
        1. Always start by understanding the text context
        2. Use the text_classifier tool for the actual classification
        3. If uncertain, use context analysis or similarity search for additional insights
        4. Provide clear reasoning for your classification decisions
        5. Consider edge cases and ambiguous content carefully
        
        Remember: You must classify text into exactly one of the provided categories.
        """
    
    async def classify_with_reasoning(self, text: str, include_analysis: bool = True) -> Dict[str, Any]:
        """Classify text with detailed reasoning and analysis."""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        start_time = time.time()
        
        try:
            # Prepare input for the agent
            agent_input = f"""
            Please classify the following text: "{text}"
            
            Available categories: {', '.join(self.labels)}
            
            {'Please also provide context analysis and reasoning.' if include_analysis else 'Focus on accurate classification.'}
            """
            
            # Run the agent
            result = await self.executor.ainvoke({
                "input": agent_input
            })
            
            # Parse the agent output
            output = result["output"]
            
            # Try to extract structured information
            classification_result = self._parse_agent_output(output, text)
            classification_result["agent_reasoning"] = output
            classification_result["processing_time"] = time.time() - start_time
            
            self.logger.info(
                "Agent classification completed",
                classification=classification_result["class"],
                confidence=classification_result["confidence"],
                processing_time=classification_result["processing_time"]
            )
            
            return classification_result
            
        except Exception as e:
            self.logger.error("Agent classification failed", error=str(e), text_length=len(text))
            raise ClassificationError(f"Agent classification failed: {str(e)}")
    
    def _parse_agent_output(self, output: str, original_text: str) -> Dict[str, Any]:
        """Parse agent output to extract classification result."""
        # Look for JSON in the output
        try:
            import re
            json_match = re.search(r'\{[^}]*"classification"[^}]*\}', output)
            if json_match:
                json_data = json.loads(json_match.group())
                return {
                    "class": json_data.get("classification", self.labels[0]),
                    "confidence": float(json_data.get("confidence", 0.5)),
                    "explanation": json_data.get("explanation", output[:200] + "..."),
                    "raw_response": output
                }
        except:
            pass
        
        # Fallback: look for classification keywords in output
        output_lower = output.lower()
        best_match = self.labels[0]
        highest_score = 0
        
        for label in self.labels:
            score = output_lower.count(label.lower())
            if score > highest_score:
                highest_score = score
                best_match = label
        
        return {
            "class": best_match,
            "confidence": 0.7 if highest_score > 0 else 0.3,
            "explanation": f"Agent reasoning: {output[:300]}...",
            "raw_response": output
        }
    
    async def classify_batch_with_reasoning(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple texts with reasoning."""
        if not texts:
            return []
        
        results = []
        for text in texts:
            try:
                result = await self.classify_with_reasoning(text, include_analysis=False)
                results.append(result)
            except Exception as e:
                self.logger.error("Batch classification failed for item", text_length=len(text), error=str(e))
                results.append({
                    "class": self.labels[0],
                    "confidence": 0.0,
                    "explanation": f"Error in batch processing: {str(e)}",
                    "error": True
                })
        
        return results
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent configuration."""
        return {
            "labels": self.labels,
            "tools_available": [tool.name for tool in self.tools],
            "memory_size": settings.AGENT_MEMORY_SIZE,
            "max_iterations": settings.AGENT_MAX_ITERATIONS,
            "model": settings.OPENAI_MODEL,
            "agent_enabled": settings.AGENT_ENABLED
        }
    
    def clear_memory(self):
        """Clear the agent's conversation memory."""
        self.memory.clear()
        self.logger.info("Agent memory cleared") 