import streamlit as st
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer

# Initialize models and database
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="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")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ChromaDB initialization
chroma_client = chromadb.PersistentClient(path="./mcq_chroma_db")
collection = chroma_client.get_or_create_collection(name="mcq_knowledge_base")

# Generate Multiple MCQs
def generate_mcqs(topic, num_questions=15):
    system_prompt = """
    You are an advanced AI-powered MCQ generator designed to create high-quality, contextually relevant multiple-choice questions.
    Your responsibilities include:
    - Generating well-structured questions related to the given topic.
    - Providing one correct answer along with three highly plausible but incorrect distractors.
    - Ensuring the questions align with the appropriate difficulty level.
    - Formatting the questions and answers clearly for easy readability.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate {num_questions} MCQs on: {topic}")
    ]
    
    try:
        response = chat.invoke(messages)
        memory.save_context({"input": topic}, {"output": response.content})
        return response.content.split('\n\n') if response else []
    except Exception as e:
        return []

# User Interface
st.set_page_config(page_title="AI MCQ Generator", layout="wide")
st.title("📚 AI-Powered MCQ Generator")
st.subheader("Generate adaptive, high-quality multiple-choice questions")

user_topic = st.text_input("Enter a topic to generate MCQs:")

if user_topic:
    mcqs = generate_mcqs(user_topic, num_questions=20)
    st.markdown("### 📝 Generated MCQs:")
    
    if not mcqs:
        st.markdown("⚠️ No questions generated. Please try again.")
    else:
        for i, mcq in enumerate(mcqs, start=1):
            lines = mcq.split('\n')
            if len(lines) < 2:
                continue  # Skip malformed MCQs
            
            question, *options = lines
            correct_answer = options[0]  # Assuming first option is the correct one
            
            st.markdown(f"**{i}. {question}**")
            for option in options:
                st.markdown(f"- {option}")
            st.markdown(f"✅ **Correct Answer:** {correct_answer}")
            st.markdown("---")
