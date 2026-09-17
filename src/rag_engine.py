"""
RAG Engine: Core Retrieval-Augmented Generation module for Restaurant Recommendations.
"""

import json
import os
import time
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

PROMPT_TEMPLATE = """You are "ConciergeAI", an expert restaurant concierge and dining recommendation assistant.

Your task is to provide helpful, personalized restaurant recommendations based strictly on the verified catalog context provided below.

RULES:
1. Recommend the best matching restaurant(s) from the provided context that fit the user's criteria (cuisine, budget, dietary restrictions, location, vibe).
2. Clearly highlight: Restaurant Name, Cuisine, Neighborhood, Price Tier, and why it fits their request. Mention a signature dish or amenity.
3. If the user asks for dietary accommodations (e.g. Vegan, Halal, Gluten-Free), explicitly confirm whether and how the restaurant caters to it.
4. If no restaurants in the context match the requested criteria (e.g. asking for a city or cuisine not in our catalog), politely state that our current catalog does not have a matching restaurant and offer the closest alternative from the provided context. DO NOT invent non-existent restaurants.

==============================
CATALOG CONTEXT:
{context}
==============================

USER QUESTION: {question}

CONCIERGE RECOMMENDATION:"""


class RestaurantRAGEngine:
    def __init__(self, dataset_path: Optional[str] = None):
        if not dataset_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dataset_path = os.path.join(base_dir, "data", "restaurants_dataset.json")
        
        self.dataset_path = dataset_path
        self.documents: List[Document] = []
        self.vector_db: Optional[Chroma] = None
        self.retriever = None
        self.llm = None
        self.models_to_try = [
            "gemini-3.8-flash",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-3.7-flash"
        ]
        self.current_model_idx = 0
        self.prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
        
        self._load_dataset()
        self._setup_vector_store()
        self._setup_llm()

    def _load_dataset(self):
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        for item in raw_data:
            content = (
                f"Restaurant Name: {item['name']}\n"
                f"Cuisine: {item['cuisine']}\n"
                f"Neighborhood: {item['neighborhood']}\n"
                f"Price Range: {item['price_range']} ({item['price_per_person']} per person)\n"
                f"Dietary Accommodations: {', '.join(item['dietary_options'])}\n"
                f"Atmosphere & Ambiance: {item['atmosphere']}\n"
                f"Signature Dishes: {', '.join(item['signature_dishes'])}\n"
                f"Features & Amenities: {', '.join(item['amenities'])}\n"
                f"Hours: {item['hours']}"
            )
            doc = Document(
                page_content=content,
                metadata={
                    "id": item["id"],
                    "name": item["name"],
                    "cuisine": item["cuisine"],
                    "neighborhood": item["neighborhood"],
                    "price": item["price_range"],
                    "rating": item.get("rating", 4.5)
                }
            )
            self.documents.append(doc)

    def _setup_vector_store(self):
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = Chroma.from_documents(
            documents=self.documents,
            embedding=embedding_model,
            collection_name="restaurant_catalog"
        )
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})

    def _setup_llm(self):
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=self.models_to_try[self.current_model_idx],
                    temperature=0.2
                )
            except Exception:
                self.llm = None

    def _format_context(self, docs: List[Document]) -> str:
        chunks = []
        for d in docs:
            name = d.metadata.get("name", "Restaurant")
            chunks.append(f"--- [CATALOG ENTRY: {name}] ---\n{d.page_content}")
        return "\n\n".join(chunks)

    def _extract_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(content)

    def _grounded_fallback(self, docs: List[Document]) -> str:
        if not docs:
            return "No matching restaurants found in catalog."
        top = docs[0]
        m = top.metadata
        return (
            f"Based on your request, I recommend **{m.get('name')}** from our verified catalog:\n\n"
            f"• **Restaurant:** {m.get('name')}\n"
            f"• **Cuisine:** {m.get('cuisine')}\n"
            f"• **Neighborhood:** {m.get('neighborhood')}\n"
            f"• **Price Tier:** {m.get('price')}\n\n"
            f"{top.page_content}\n\n"
            f"*(Generated from ChromaDB retrieved context)*"
        )

    def query(self, user_query: str, top_k: int = 3) -> Dict[str, Any]:
        retrieved_docs = self.retriever.invoke(user_query)
        context_str = self._format_context(retrieved_docs)
        
        answer = None
        if self.llm:
            formatted_prompt = self.prompt.format(context=context_str, question=user_query)
            for idx in range(len(self.models_to_try)):
                model_name = self.models_to_try[(self.current_model_idx + idx) % len(self.models_to_try)]
                try:
                    active_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
                    response = active_llm.invoke(formatted_prompt)
                    self.current_model_idx = (self.current_model_idx + idx) % len(self.models_to_try)
                    self.llm = active_llm
                    answer = self._extract_text(response.content)
                    break
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        continue
                    elif "503" in err_msg or "UNAVAILABLE" in err_msg:
                        time.sleep(2)
                        continue
                    else:
                        continue
        
        if not answer:
            answer = self._grounded_fallback(retrieved_docs)

        return {
            "query": user_query,
            "retrieved_documents": retrieved_docs,
            "answer": answer
        }
