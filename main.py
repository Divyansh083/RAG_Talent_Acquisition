import os
import json
from groq import Groq
import moviepy.editor as mp
from PyPDF2 import PdfReader
from pydub import AudioSegment
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CURRENT_FOLDER = os.path.abspath(os.getcwd())
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
GROQ_MODEL = "llama3-8b-8192"
TRANSCRIPTION_MODEL = "whisper-large-v3"
MAX_AUDIO_SIZE_MB = 24
AUDIO_CHUNK_MINUTES = 12
VECTORSTORE_PATH = os.path.join(os.path.join(CURRENT_FOLDER, "vector_store"), "candidate_vectorstore.faiss")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def extract_candidate_info(file_path):
    filename = os.path.basename(file_path)
    parts = filename.split("_")
    if len(parts) >= 4:
        return {
            "candidate_name": parts[0].replace("-", " "),
            "source_type": "resume" if file_path.endswith(".pdf") else "interview"
        }
    return {
        "candidate_name": "unknown",
        "source_type": "resume" if file_path.endswith(".pdf") else "interview"
    }

def process_resume(file_path):
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += (page.extract_text() or "") + "\n"
        
        metadata = extract_candidate_info(file_path)
        print(f"Processing resume: {file_path}")
        print(f"Metadata: {json.dumps(metadata, indent=2)}")
        
        return Document(
            page_content=text.strip(),
            metadata=metadata
        )
    except Exception as e:
        print(f"Error processing resume {file_path}: {str(e)}")
        return None

def extract_audio_from_video(video_path):
    try:
        output_dir = os.path.join(CURRENT_FOLDER, "extracted_audio")
        ensure_dir(output_dir)
        
        base_name = os.path.basename(video_path)
        if base_name.lower().endswith('.mp4'):
            base_name = base_name[:-4]
        
        output_path = os.path.join(output_dir, f"{base_name}.ogg")
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(output_path, codec='libvorbis')
        
        print(f"Extracted audio to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return None

def split_audio_file(audio_path, chunk_minutes=AUDIO_CHUNK_MINUTES):
    try:
        audio = AudioSegment.from_file(audio_path)
        chunk_ms = chunk_minutes * 60 * 1000
        chunks = []

        output_dir = os.path.join(CURRENT_FOLDER, "audio_chunk")
        ensure_dir(output_dir)

        for i, start in enumerate(range(0, len(audio), chunk_ms)):
            end = start + chunk_ms
            chunk = audio[start:end]
            chunk_filename = os.path.join(output_dir, f"audio_chunk_{i}.ogg")
            chunk.export(chunk_filename, format="ogg")
            chunks.append(chunk_filename)

        return chunks
    except Exception as e:
        print(f"Error splitting audio: {str(e)}")
        return []

def process_interview_video(file_path):
    try:
        print(f"Processing interview video: {file_path}")
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        transcript_dir = os.path.join(CURRENT_FOLDER, "transcripts")
        ensure_dir(transcript_dir)
        transcript_path = os.path.join(transcript_dir, f"{base_name}.txt")

        # Load existing transcript if present
        if os.path.exists(transcript_path):
            print(f"Loading cached transcript from {transcript_path}")
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = f.read()
        else:
            # Transcribe audio
            audio_path = extract_audio_from_video(file_path)
            if not audio_path:
                return None

            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            if file_size_mb <= MAX_AUDIO_SIZE_MB:
                print(f"Transcribing audio file: {audio_path}")
                with open(audio_path, "rb") as audio_file:
                    transcription = groq_client.audio.transcriptions.create(
                        file=(os.path.basename(audio_path), audio_file.read()),
                        model=TRANSCRIPTION_MODEL,
                        response_format="text"
                    )
                transcript = transcription
            else:
                print(f"Splitting and transcribing large audio file: {audio_path}")
                chunks = split_audio_file(audio_path)
                full_transcript = []
                for chunk_path in chunks:
                    with open(chunk_path, "rb") as chunk_file:
                        transcription = groq_client.audio.transcriptions.create(
                            file=(os.path.basename(chunk_path), chunk_file.read()),
                            model=TRANSCRIPTION_MODEL,
                            response_format="text"
                        )
                        full_transcript.append(transcription)
                    os.unlink(chunk_path)
                transcript = "\n".join(full_transcript)

            os.unlink(audio_path)

            # Save transcript
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)

        metadata = extract_candidate_info(file_path)
        print(f"Transcript metadata: {json.dumps(metadata, indent=2)}")

        return Document(
            page_content=transcript,
            metadata=metadata
        )
    except Exception as e:
        print(f"Error processing video {file_path}: {str(e)}")
        return None

def get_all_documents():
    documents = []
    data_path = os.path.join(CURRENT_FOLDER, "dataset")
    if not os.path.exists(data_path):
        print("Data directory not found!")
        return documents

    for candidate_folder in os.listdir(data_path):
        candidate_path = os.path.join(data_path, candidate_folder)
        if not os.path.isdir(candidate_path):
            continue
        
        try:
            print(f"\nProcessing candidate folder: {candidate_folder}")
            for filename in os.listdir(candidate_path):
                file_path = os.path.join(candidate_path, filename)
                if filename.endswith(".pdf"):
                    doc = process_resume(file_path)
                    if doc:
                        documents.append(doc)
                elif filename.lower().endswith((".mp4", ".m4a", ".mp3", ".wav", ".ogg")):
                    doc = process_interview_video(file_path)
                    if doc:
                        documents.append(doc)
        except Exception as e:
            print(f"Error processing candidate {candidate_folder}: {str(e)}")
            continue
    
    print(f"\nTotal documents processed: {len(documents)}")
    return documents

def get_text_chunks(documents):
    if not documents:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    
    chunks = text_splitter.create_documents(texts, metadatas=metadatas)
    print(f"\nSplit documents into {len(chunks)} chunks")
    return chunks

def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    if os.path.exists(VECTORSTORE_PATH):
        print("Loading existing vectorstore...")
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new vectorstore...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

def extract_metadata_from_query(query, llm):
    prompt = (
        """
        you are a highly accurate data processing assistant. your task is to 
        generate a JSON schema response that strictly adheres to the provided JSON 
        schema, regardless of the input provided. Always include all the fields 
        specified in the schema, using appropriate empty values (e.g., '') if data 
        is missing or unclear. The response must be a valid JSON with proper syntax,
        escaping and formatting. Below is the JSON schema with hints for each field,
        followed by instructions.
        """
        """
        ### JSON Schema

        {
            "candidates_info": [
                {
                    "candidate_name": "",
                    "source_type": ""
                }
            ]
        }
        """
        """
        candidate_name: full name including first and last name
        source_type: required, must be 'resume' or 'interview'. Use 'resume' if the query is about technologies, skills, or certifications. Use 'interview' otherwise.
        """
        """
        
        ### Instructions
        1. Adhere to Schema: Generate a JSON object that strictly matches the 
        schema exactly, including all required fields and nested structures.
        2. Handle Missing Data: If input data is incomplete, ambiguous, or missing, 
        use the following defaults: ''
        3. Use Hints: follow the description comments after #HINT symbol for each 
        field for expected formats and examples
        4. Mandatory Sections: Select a value from the choices if provided. 
        You must generate a value for fields marked as #HINT #REQUIRED.
        is mandatory. 
        5. Case sensitivity: Strictly use lower case for all values.
        6. Valid JSON: Ensure the response is properly formatted JSON with correct 
        syntax, escaping, and no trailing commas.
        7. Single response: Return only the JSON object, enclosed in triple backticks
        (```json), with no additional text or explanations.
        
        """
        f"""
        ### Input
        {query}
    
    """)

    try:
        response = llm.invoke(prompt)
        json_str = response.content.strip()

        if json_str.startswith("```json"):
            json_str = json_str[7:-3].strip()
        elif json_str.startswith("```"):
            json_str = json_str[3:-3].strip()

        if not json_str:
            raise ValueError("Empty JSON string")

        metadata = json.loads(json_str.lower())

        if not isinstance(metadata, dict) or "candidates_info" not in metadata:
            raise ValueError("Invalid structure")

        return metadata

    except Exception as e:
        print(f"Metadata extraction failed. Raw response: '{json_str}'. Error: {str(e)}")
        return {
            "candidates_info": [{
                "candidate_name": "",
                "source_type": ""
            }]
        }

def create_faiss_filter(metadata):
    """Create filter with validation"""
    if not metadata or not isinstance(metadata.get("candidates_info"), list):
        return None
    
    candidate_info = metadata["candidates_info"][0] if metadata["candidates_info"] else {}
    filter_dict = {}
    
    # Only add filters if values exist
    if candidate_info.get("candidate_name"):
        filter_dict["candidate_name"] = candidate_info["candidate_name"].lower().strip()
    if candidate_info.get("source_type") in ["resume", "interview"]:
        filter_dict["source_type"] = candidate_info["source_type"]
    
    print("FAISS filter:", filter_dict)
    return filter_dict if filter_dict else None

def get_conversation_chain(vectorstore):
    if not vectorstore:
        return None
    
    try:
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=GROQ_MODEL,
            temperature=0
        )

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={
                    "k": 5,
                    "filter": None
                }
            ),
            return_source_documents=True
        )
    except Exception as e:
        print(f"Error creating conversation chain: {str(e)}")
        return None

def query_system(qa, query, llm):
    try:
        print(f"\nProcessing query: '{query}'")
        
        metadata = extract_metadata_from_query(query, llm)
        faiss_filter = create_faiss_filter(metadata)
        
        qa.retriever.search_kwargs["filter"] = faiss_filter
        print("Search kwargs:", qa.retriever.search_kwargs)
        
        response = qa({"query": query})
        
        if "source_documents" in response:
            print("\nMatched Documents:")
            for i, doc in enumerate(response["source_documents"]):
                print(f"{i+1}. {doc.metadata.get('candidate_name','Unknown')} "
                      f"({doc.metadata.get('source_type','unknown')})")
                print(f"   Content: {doc.page_content[:100]}...")
        
        return response["result"]
    
    except Exception as e:
        print(f"Query failed: {str(e)}")
        return "I couldn't process that request. Please try rephrasing your question."

def process_files():
    if os.path.exists(VECTORSTORE_PATH):
        vectorstore = get_vectorstore(chunks=None)
        conversation = get_conversation_chain(vectorstore)
        print("Loaded existing vectorstore!")
    else:
        documents = get_all_documents()
        if not documents:
            print("No candidate data was loaded.")
            return None
        else:
            chunks = get_text_chunks(documents)
            if chunks:
                vectorstore = get_vectorstore(chunks)
                conversation = get_conversation_chain(vectorstore)
                print("Successfully processed all candidate data!")
            else:
                print("No valid text chunks were created.")
                return None
    return conversation

def main():
    conversation = process_files()
    if not conversation:
        return
    
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=GROQ_MODEL,
        temperature=0
    )
    
    print("\nCandidate Interview Analysis System")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("Ask a question about the candidates: ")
        if query.lower() == 'exit':
            break
        
        if query and conversation:
            print("Generating answer...")
            result = query_system(conversation, query, llm)
            print("\nAnswer:")
            print(result)
            print()

if __name__ == '__main__':
    main()