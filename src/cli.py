"""
Command-line interface for the Restaurant Recommendation RAG Bot.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine import RestaurantRAGEngine

def main():
    print("=" * 70)
    print("🍽️ Welcome to ConciergeAI (Restaurant Recommendation RAG Bot)")
    print("Type your question below (or 'exit' / 'quit' to stop).")
    print("=" * 70)
    
    engine = RestaurantRAGEngine()
    print("✅ Ready! Ask anything (cuisine, dietary needs, budget, occasion)...")

    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Thank you for using ConciergeAI! Bon appétit!")
                break
            
            result = engine.query(user_input)
            
            print("\n📑 Retrieved from Vector DB:")
            for i, doc in enumerate(result["retrieved_documents"], 1):
                m = doc.metadata
                print(f"   [{i}] {m.get('name')} | {m.get('cuisine')} | {m.get('neighborhood')} | {m.get('price')}")
            
            print("\n🤖 ConciergeAI:")
            print(result["answer"])
            print("-" * 70)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
