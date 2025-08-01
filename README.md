# 🧠 ThinkPDF - AI-Powered Document Intelligence Platform

ThinkPDF is a Streamlit-based web application that allows you to upload PDF documents and interact with them using AI. Ask questions about your documents and get intelligent responses powered by Google's Gemini AI model.

## ✨ Features

- 📄 **Multi-PDF Upload**: Upload and process multiple PDF files simultaneously
- 🤖 **AI-Powered Chat**: Ask questions about your documents using natural language
- 🔍 **Semantic Search**: Find relevant information across all your uploaded documents
- 📊 **Source Attribution**: View the exact document sections used to generate responses
- ⚡ **Quick Actions**: Pre-built buttons for summaries, key points, and topic extraction
- 🎨 **Modern UI**: Clean, responsive interface with custom styling
- 💾 **Persistent Storage**: Documents are processed once and stored for future queries

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google API Key (for Gemini AI and embeddings)

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd ThinkPDF
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 🔧 Configuration

### Google API Setup

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add the key to your `.env` file

### Model Configuration

The application uses:
- **Embeddings**: `models/embedding-001`
- **Chat Model**: `gemini-2.5-flash` with temperature 0.3

You can modify these in the source code if needed.

## 📖 How to Use

### 1. Upload Documents
- Use the sidebar "Document Manager"
- Click "Upload PDF Files" and select your PDFs
- View file details (name and size)

### 2. Process Documents
- Click "⚡ Process Documents" to extract text and create embeddings
- Wait for processing to complete (progress bar shows status)

### 3. Ask Questions
- Type your question in the main chat interface
- Or use quick action buttons:
  - 📋 **Summarize**: Get a summary of main content
  - 🔍 **Key Points**: Extract important findings
  - 📊 **Topics**: Identify main discussion topics

### 4. View Results
- AI responses are displayed in a styled container
- Expand "📄 View Source Documents" to see relevant excerpts
- Source attribution shows which documents were used

## 🏗️ Architecture

### Core Components

```
├── PDF Processing
│   ├── get_pdf_text()      # Extract text from PDFs
│   ├── get_text_chunks()   # Split text into manageable chunks
│   └── get_vector_store()  # Create and save embeddings
│
├── AI Processing
│   ├── get_conversational_chain()  # Setup Q&A chain
│   └── user_input()               # Handle user queries
│
└── User Interface
    └── main()              # Streamlit app layout
```

### Technology Stack

- **Frontend**: Streamlit
- **PDF Processing**: PyPDF2
- **AI Framework**: LangChain
- **Vector Database**: FAISS
- **AI Models**: Google Gemini & Embeddings
- **Text Processing**: RecursiveCharacterTextSplitter

### Data Flow

1. **Upload** → PDFs uploaded via Streamlit file uploader
2. **Extract** → Text extracted using PyPDF2
3. **Chunk** → Text split into 10,000 character chunks (1,000 overlap)
4. **Embed** → Google embeddings created and stored in FAISS
5. **Query** → User questions processed through semantic search
6. **Generate** → Gemini AI generates responses from relevant chunks

## 📁 Project Structure

```
ThinkPDF/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
├── faiss_index/       # Vector database (auto-generated)
└── README.md          # This file
```

## 🛠️ Dependencies

```
streamlit              # Web framework
PyPDF2                # PDF text extraction
langchain             # AI workflow orchestration
langchain-google-genai # Google AI integration
google-generativeai   # Google Gemini API
faiss-cpu             # Vector similarity search
python-dotenv         # Environment variable management
```

## 🔒 Security Notes

- Keep your `GOOGLE_API_KEY` secure and never commit it to version control
- The application uses `allow_dangerous_deserialization=True` for FAISS - only use with trusted data
- Uploaded files are processed locally and not sent to external services except for AI processing

## 🐛 Troubleshooting

### Common Issues

1. **"No documents found" error**
   - Make sure you've uploaded and processed PDFs first
   - Check if `faiss_index` folder exists

2. **API Key errors**
   - Verify your Google API key is correct
   - Ensure the key has access to Gemini and embedding models

3. **Processing errors**
   - Check PDF file integrity
   - Ensure sufficient disk space for embeddings

4. **Slow processing**
   - Large PDFs take time to process
   - Consider splitting very large documents

### Clear Data

Use the "🗑️ Clear Data" button in the sidebar to reset the vector database.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.

## 🆘 Support

For issues and questions:
- Check the troubleshooting section above
- Review error messages in the Streamlit interface
- Ensure all dependencies are properly installed

---

**Built with ❤️ using Streamlit and Google AI**
