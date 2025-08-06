from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import google.generativeai as genai
import httpx
import tempfile
import os
import time
import asyncio
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Q&A API",
    description="Process documents and answer questions using Gemini AI with parallel processing and enhanced reasoning",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
VALID_TOKEN = "8b19aa4e64ea7d5aa15448e401460637d5d9ba07e3a839ae961d745fb0910de3"
GEMINI_API_KEY = "AIzaSyB8pAIHQ6uMWRgz_a1x_73JIK9jJTw0SQ8"

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Thread-safe model initialization
_thread_local = threading.local()

def get_model():
    """Get thread-local Gemini model instance"""
    if not hasattr(_thread_local, 'model'):
        _thread_local.model = genai.GenerativeModel('gemini-1.5-pro')
    return _thread_local.model

# Pydantic models
class DocumentRequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]

class DocumentResponse(BaseModel):
    answers: List[str]

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != VALID_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return credentials.credentials

class GeminiDocumentProcessor:
    def __init__(self, max_concurrent_questions: int = 10):
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.uploaded_files = {}
        self.max_concurrent_questions = max_concurrent_questions
        # Semaphore to limit concurrent API calls to avoid rate limiting
        self.semaphore = asyncio.Semaphore(max_concurrent_questions)
    
    async def download_document(self, url: str) -> str:
        """Download document from URL and save to temp file"""
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Create temp file
                suffix = Path(url.split('?')[0]).suffix or '.pdf'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(response.content)
                    return tmp.name
                    
        except Exception as e:
            logger.error(f"Error downloading document: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
    
    async def upload_to_gemini(self, file_path: str) -> str:
        """Upload document to Gemini File API"""
        try:
            logger.info(f"Uploading {file_path} to Gemini...")
            
            # Upload file
            file = genai.upload_file(
                path=file_path,
                display_name=os.path.basename(file_path)
            )
            
            # Wait for processing
            max_wait = 300  # 5 minutes timeout
            wait_time = 0
            
            while file.state.name == "PROCESSING" and wait_time < max_wait:
                logger.info("Document processing...")
                await asyncio.sleep(2)
                wait_time += 2
                file = genai.get_file(file.name)
            
            if file.state.name == "FAILED":
                raise Exception(f"File processing failed: {file.state.name}")
            
            if wait_time >= max_wait:
                raise Exception("Document processing timeout")
                
            logger.info(f"Successfully uploaded: {file.display_name}")
            return file
            
        except Exception as e:
            logger.error(f"Gemini upload failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
        finally:
            # Cleanup temp file
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def create_enhanced_prompt(self, question: str) -> str:
        """Create an enhanced prompt for better reasoning and detailed responses"""
        return f"""
You are an expert document analyst with exceptional reasoning abilities. Your task is to provide comprehensive, well-reasoned answers based on the uploaded document.

QUESTION TO ANALYZE: {question}

REASONING FRAMEWORK - Follow this systematic approach:

1. DOCUMENT COMPREHENSION:
   - First, carefully read and understand the entire document context
   - Identify all sections, chapters, or parts that might be relevant to the question
   - Note the document's structure, purpose, and key themes

2. QUESTION ANALYSIS:
   - Break down the question into its core components
   - Identify what type of information is being requested (factual, analytical, comparative, etc.)
   - Determine the scope and specificity required in the answer

3. INFORMATION EXTRACTION:
   - Systematically search through the document for relevant information
   - Identify direct statements that address the question
   - Look for implied or contextual information that supports the answer
   - Note any related information that provides additional context

4. REASONING AND SYNTHESIS:
   - Connect different pieces of information from various parts of the document
   - Analyze relationships between concepts, facts, and ideas
   - Consider the broader context and implications
   - Identify patterns, trends, or themes that emerge

5. RESPONSE CONSTRUCTION:
   Follow this structure for your detailed answer:

   **DIRECT ANSWER:** Start with a clear, direct response to the question.

   **DETAILED EXPLANATION:** Provide a comprehensive explanation that includes:
   - Relevant facts, figures, dates, names, and specific details from the document
   - Context and background information that helps understand the answer
   - Multiple perspectives or aspects if applicable
   - Connections between different parts of the document

   **SUPPORTING EVIDENCE:** Reference specific sections, quotes, or data points from the document that support your answer.

   **ADDITIONAL CONTEXT:** Include any relevant supplementary information that enhances understanding of the topic.

   **IMPLICATIONS/SIGNIFICANCE:** If applicable, explain why this information is important or what it means in the broader context.

QUALITY STANDARDS:
- Be thorough and comprehensive while staying relevant to the question
- Use specific details, numbers, dates, percentages, and concrete examples from the document
- Explain technical terms or concepts that might need clarification
- Show your reasoning process - explain how you arrived at your conclusions
- If information spans multiple sections, synthesize it coherently
- Maintain accuracy - only include information that is explicitly stated or clearly implied in the document

ERROR HANDLING:
- If the information is not found in the document, clearly state: "The requested information is not available in the provided document."
- If information is partially available, explain what is available and what is missing
- If there are contradictions in the document, acknowledge and explain them
- If the question requires information beyond what's in the document, clarify the limitations

RESPONSE TONE:
- Professional and authoritative
- Clear and well-structured
- Detailed but not unnecessarily verbose
- Accessible to the intended audience

Now, apply this framework to answer the question thoroughly and provide a detailed, well-reasoned response based solely on the document content.
"""

    async def answer_single_question(self, gemini_file, question: str, question_index: int) -> tuple[int, str]:
        """Answer a single question about the document with enhanced reasoning and rate limiting"""
        async with self.semaphore:
            try:
                logger.info(f"Processing question {question_index} with enhanced reasoning: {question[:50]}...")
                
                # Use the enhanced prompt
                enhanced_prompt = self.create_enhanced_prompt(question)
                
                # Use thread-local model to avoid threading issues
                loop = asyncio.get_event_loop()
                
                def generate_content():
                    model = get_model()
                    return model.generate_content([enhanced_prompt, gemini_file])
                
                # Run in thread pool to avoid blocking
                with ThreadPoolExecutor(max_workers=1) as executor:
                    response = await loop.run_in_executor(executor, generate_content)
                
                answer = response.text.strip()
                logger.info(f"Completed enhanced processing for question {question_index}")
                
                return question_index, answer
                
            except Exception as e:
                logger.error(f"Error answering question {question_index}: {str(e)}")
                return question_index, f"Error processing question: {str(e)}"
    
    async def answer_questions_parallel(self, gemini_file, questions: List[str]) -> List[str]:
        """Answer multiple questions about the document in parallel with enhanced reasoning"""
        logger.info(f"Starting enhanced parallel processing of {len(questions)} questions...")
        start_time = time.time()
        
        # Create tasks for all questions
        tasks = [
            self.answer_single_question(gemini_file, question, i + 1) 
            for i, question in enumerate(questions)
        ]
        
        # Execute all tasks concurrently with progress tracking
        completed_tasks = []
        try:
            # Use asyncio.as_completed to process results as they come in
            for coro in asyncio.as_completed(tasks):
                try:
                    question_index, answer = await coro
                    completed_tasks.append((question_index, answer))
                    logger.info(f"Enhanced processing progress: {len(completed_tasks)}/{len(questions)} questions completed")
                except Exception as e:
                    logger.error(f"Task failed: {str(e)}")
                    completed_tasks.append((len(completed_tasks) + 1, f"Error: {str(e)}"))
        
        except Exception as e:
            logger.error(f"Error in parallel processing: {str(e)}")
            # If something goes wrong, fall back to sequential processing
            return await self.answer_questions_sequential(gemini_file, questions)
        
        # Sort answers by question index to maintain order
        completed_tasks.sort(key=lambda x: x[0])
        answers = [answer for _, answer in completed_tasks]
        
        processing_time = time.time() - start_time
        logger.info(f"Completed all {len(questions)} questions with enhanced reasoning in {processing_time:.2f} seconds")
        logger.info(f"Average time per question: {processing_time/len(questions):.2f} seconds")
        
        return answers
    
    async def answer_questions_sequential(self, gemini_file, questions: List[str]) -> List[str]:
        """Fallback sequential processing method with enhanced reasoning"""
        logger.info("Using enhanced sequential processing as fallback")
        answers = []
        
        for i, question in enumerate(questions, 1):
            try:
                logger.info(f"Processing question {i}/{len(questions)} with enhanced reasoning: {question[:50]}...")
                
                # Use the enhanced prompt
                enhanced_prompt = self.create_enhanced_prompt(question)
                
                response = self.model.generate_content([enhanced_prompt, gemini_file])
                answer = response.text.strip()
                answers.append(answer)
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error answering question {i}: {str(e)}")
                answers.append(f"Error processing question: {str(e)}")
        
        return answers
    
    async def cleanup_gemini_file(self, gemini_file):
        """Delete file from Gemini"""
        try:
            genai.delete_file(gemini_file.name)
            logger.info(f"Cleaned up Gemini file: {gemini_file.display_name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup Gemini file: {str(e)}")

# Initialize processor with configurable concurrency
processor = GeminiDocumentProcessor(max_concurrent_questions=8)  # Adjust based on API limits

@app.get("/")
async def root():
    return {
        "message": "Enhanced Document Q&A API with Advanced Reasoning and Parallel Processing", 
        "version": "2.0.0",
        "features": ["enhanced_reasoning", "detailed_responses", "parallel_processing", "rate_limiting", "thread_safe"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "gemini_configured": bool(GEMINI_API_KEY),
        "max_concurrent_questions": processor.max_concurrent_questions,
        "enhanced_reasoning": True
    }

@app.post("/api/v1/hackrx/run", response_model=DocumentResponse)
async def process_document(
    request: DocumentRequest,
    token: str = Depends(verify_token)
):
    """
    Process a document and answer questions about it using enhanced reasoning and parallel processing
    
    - **documents**: URL to the document (PDF, DOCX, TXT)
    - **questions**: List of questions to answer about the document (processed in parallel with enhanced reasoning)
    
    The enhanced system provides:
    - Comprehensive document analysis
    - Detailed, well-reasoned responses
    - Structured answers with supporting evidence
    - Context and implications
    - Professional, thorough explanations
    """
    
    try:
        logger.info(f"Processing document with enhanced reasoning: {request.documents}")
        logger.info(f"Number of questions: {len(request.questions)}")
        
        # Download document
        temp_file_path = await processor.download_document(request.documents)
        
        # Upload to Gemini
        gemini_file = await processor.upload_to_gemini(temp_file_path)
        
        try:
            # Answer questions in parallel with enhanced reasoning
            answers = await processor.answer_questions_parallel(gemini_file, request.questions)
            
            return DocumentResponse(answers=answers)
            
        finally:
            # Always cleanup Gemini file
            await processor.cleanup_gemini_file(gemini_file)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/api/v1/hackrx/run/batch", response_model=List[DocumentResponse])
async def process_multiple_documents(
    requests: List[DocumentRequest],
    token: str = Depends(verify_token)
):
    """
    Process multiple documents in batch with enhanced reasoning
    (each document's questions are processed in parallel with detailed analysis)
    """
    
    results = []
    
    for i, request in enumerate(requests, 1):
        try:
            logger.info(f"Processing batch item {i}/{len(requests)} with enhanced reasoning")
            result = await process_document(request, token)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing batch item {i}: {str(e)}")
            results.append(DocumentResponse(answers=[f"Error: {str(e)}"]))
    
    return results

@app.get("/api/v1/hackrx/status")
async def get_status(token: str = Depends(verify_token)):
    """Get API status and configuration"""
    return {
        "status": "operational",
        "version": "2.0.0",
        "gemini_model": "gemini-1.5-pro",
        "max_file_size": "2GB",
        "supported_formats": ["PDF", "DOCX", "TXT"],
        "authentication": "Bearer token required",
        "parallel_processing": True,
        "enhanced_reasoning": True,
        "max_concurrent_questions": processor.max_concurrent_questions,
        "features": {
            "enhanced_prompting": "Systematic reasoning framework with detailed analysis",
            "structured_responses": "Direct answer, explanation, evidence, context, and implications",
            "comprehensive_analysis": "Multi-step reasoning process for thorough understanding",
            "rate_limiting": "Built-in semaphore control",
            "thread_safety": "Thread-local model instances",
            "error_handling": "Graceful fallback to sequential processing",
            "progress_tracking": "Real-time completion logging"
        },
        "reasoning_framework": {
            "document_comprehension": "Full document understanding and structure analysis",
            "question_analysis": "Component breakdown and scope determination",
            "information_extraction": "Systematic search and context identification",
            "reasoning_synthesis": "Connection of concepts and pattern recognition",
            "response_construction": "Structured, detailed answer formatting"
        }
    }

@app.put("/api/v1/hackrx/config/concurrency")
async def update_concurrency(
    max_concurrent: int,
    token: str = Depends(verify_token)
):
    """Update maximum concurrent question processing limit"""
    if max_concurrent < 1 or max_concurrent > 20:
        raise HTTPException(
            status_code=400,
            detail="Concurrency limit must be between 1 and 20"
        )
    
    processor.max_concurrent_questions = max_concurrent
    processor.semaphore = asyncio.Semaphore(max_concurrent)
    
    return {
        "message": f"Concurrency limit updated to {max_concurrent}",
        "max_concurrent_questions": processor.max_concurrent_questions,
        "enhanced_reasoning": "Enabled for all concurrent processing"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "details": str(exc)
    }

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    HOST = "0.0.0.0"
    PORT = 8000
    
    logger.info(f"Starting enhanced server with advanced reasoning on {HOST}:{PORT}")
    logger.info(f"Max concurrent questions: {processor.max_concurrent_questions}")
    logger.info(f"Enhanced reasoning: Enabled")
    logger.info(f"API Base URL: http://localhost:{PORT}/api/v1")
    logger.info(f"Documentation: http://localhost:{PORT}/docs")
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )
