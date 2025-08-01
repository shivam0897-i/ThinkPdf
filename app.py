import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        if not os.path.exists("faiss_index"):
            st.error("❌ No documents found! Please upload and process PDF files first.")
            return
        
        with st.spinner("🔍 Searching your documents..."):
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        # response display
        st.markdown("---")
        st.markdown("### 🤖 AI Response:")
        st.markdown(f"""
    <div style="background-color: #e3f2fd;
                padding: 1rem;
                border-radius: 10px;
                border-left: 4px solid #2196f3;
                margin: 1rem 0;
                color: #0d47a1;">
        {response["output_text"]}
    </div>
""", unsafe_allow_html=True)
        
        # Source documents
        with st.expander("📄 View Source Documents", expanded=False):
            for i, doc in enumerate(docs):
                st.markdown(f"**Source {i+1}:**")
                st.text_area(f"Content {i+1}:", doc.page_content[:500] + "...", height=100, disabled=True)
                
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        st.error(f"❌ Error: {str(e)}")
        st.error(f"❌ Error: {str(e)}")


def main():
    # page config
    st.set_page_config(
        page_title="ThinkPDF - AI Chat",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS
    st.markdown("""
<style>
.main-header { text-align: center; padding: 0.8rem 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
border-radius: 10px; margin-bottom: 1.2rem; color: white; font-size: 1.5rem; box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
                
.stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 16px;
 padding: 0.5rem 1.2rem; font-weight: 600; font-size: 0.95rem; transition: all 0.25s ease; }
                
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
                
.sidebar-content { background: #f8f9fa; padding: 1rem; border-radius: 8px; border: 1px solid #dee2e6; font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)
    
    # Modern header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 ThinkPDF</h1>
        <p>AI-Powered Document Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("## 💬 Chat with Your Documents")
        
        # Input with examples
        user_question = st.text_input(
            "Ask a question:",
            placeholder="What are the main points discussed in the documents?",
            help="Ask specific questions for better results"
        )
        
        # buttons

        with col2:
            if st.button("📋 Summarize", key="sum"):
                user_question = "Please provide a summary of the main content"
        with col2:
            if st.button("🔍 Key Points", key="key"):
                user_question = "What are the key findings or important points?"
        with col2:
            if st.button("📊 Topics", key="top"):
                user_question = "What are the main topics discussed?"
        
        if user_question:
            user_input(user_question)

    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Document Manager")
        
        pdf_docs = st.file_uploader(
            "Upload PDF Files",
            accept_multiple_files=True,
            type=['pdf'],
            help="Select one or more PDF files"
        )
        
        if pdf_docs:
            st.success(f"📄 {len(pdf_docs)} file(s) selected")
            for i, pdf in enumerate(pdf_docs, 1):
                file_size = len(pdf.getvalue()) // 1024
                st.write(f"{i}. `{pdf.name}` ({file_size} KB)")
        
        if st.button("⚡ Process Documents", use_container_width=True):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.toast("📁 Documents processed successfully!")     
                    except (ValueError, FileNotFoundError, RuntimeError) as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("Please upload PDF files first!")
        # Quick actions
        st.markdown("---")
        if st.button("🗑️ Clear Data"):
            if os.path.exists("faiss_index"):
                import shutil
                shutil.rmtree("faiss_index")
                st.success("✅ Data cleared!")
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
