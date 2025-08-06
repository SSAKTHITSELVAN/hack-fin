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
    description="Process documents and answer questions using Gemini AI with parallel processing",
    version="1.0.0"
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
        _thread_local.model = genai.GenerativeModel('gemini-2.5-flash')
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
        self.model = genai.GenerativeModel('gemini-2.5-flash')
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
        You are an expert document analyst with exceptional reading comprehension and analytical skills. Your task is to provide comprehensive, well-reasoned answers based on the uploaded document.

        QUESTION: {question}

        ANALYSIS FRAMEWORK:
        Please follow this structured approach to answer the question:

        1. CONTEXT UNDERSTANDING: First, carefully read and understand the entire document context
        2. INFORMATION EXTRACTION: Identify all relevant information that relates to the question
        3. LOGICAL REASONING: Connect different pieces of information and analyze their relationships
        4. COMPREHENSIVE SYNTHESIS: Combine findings into a coherent, detailed explanation
        5. VERIFICATION: Cross-reference information to ensure accuracy and completeness

        ENHANCED REASONING WITH GEMINI 2.5 CAPABILITIES:
        - Utilize the advanced reasoning capabilities of Gemini 2.5 Flash
        - Apply multi-step logical thinking to complex document relationships
        - Leverage improved contextual understanding for nuanced interpretations
        - Use enhanced analytical processing for comprehensive document synthesis

        RESPONSE REQUIREMENTS:
        - Provide a detailed, elaborate explanation with proper reasoning chains
        - Include specific evidence, data points, numbers, dates, percentages, and examples from the document
        - Explain the context and significance of the information with deeper analysis
        - Show logical connections between different pieces of information across document sections
        - Address multiple aspects and implications of the question comprehensively
        - Use clear, professional language with well-structured argumentation
        - CRITICAL: Your response must be exactly 5 lines long - no more, no less
        - Each line should contain substantial information and contribute to a complete, detailed answer
        - Make every line count by packing detailed information while maintaining readability and flow

        CRITICAL INSTRUCTIONS:
        - Base your answer ONLY on information explicitly stated in the document
        - If specific information is not found, state "Information not found in the document" but still provide context about what IS available
        - Do not make assumptions or add external knowledge not present in the document
        - Prioritize accuracy and completeness over brevity while respecting the 5-line constraint
        - Ensure each of the 5 lines provides meaningful, detailed information with clear reasoning
        - Connect related information across different sections of the document when relevant
        - Use Gemini 2.5's enhanced thinking capabilities for deeper document analysis

        FORMAT YOUR RESPONSE:
        Line 1: [Direct answer with key information, context, and primary evidence]
        Line 2: [Supporting details, specific data points, metrics, and corroborating evidence]
        Line 3: [Additional relevant information, broader context, and cross-referenced details]
        Line 4: [Implications, significance, relationships, and analytical insights]
        Line 5: [Comprehensive summary, conclusions, and final important contextual details]
        """
    
    async def answer_single_question(self, gemini_file, question: str, question_index: int) -> tuple[int, str]:
        """Answer a single question about the document with rate limiting"""
        async with self.semaphore:
            try:
                logger.info(f"Processing question {question_index}: {question[:50]}...")
                
                # Use the enhanced prompt
                prompt = self.create_enhanced_prompt(question)
                
                # Use thread-local model to avoid threading issues
                loop = asyncio.get_event_loop()
                
                def generate_content():
                    model = get_model()
                    return model.generate_content([prompt, gemini_file])
                
                # Run in thread pool to avoid blocking
                with ThreadPoolExecutor(max_workers=1) as executor:
                    response = await loop.run_in_executor(executor, generate_content)
                
                answer = response.text.strip()
                
                # Ensure the response follows the 5-line format
                lines = answer.split('\n')
                lines = [line.strip() for line in lines if line.strip()]
                
                # If response doesn't have exactly 5 lines, format it properly
                if len(lines) != 5:
                    # Try to reformat into 5 substantial lines
                    if len(lines) < 5:
                        # Pad with context if needed
                        while len(lines) < 5:
                            lines.append("Additional context may be limited in the source document.")
                    else:
                        # Consolidate into 5 lines
                        lines = lines[:5]
                
                formatted_answer = '\n'.join(lines)
                logger.info(f"Completed question {question_index}")
                
                return question_index, formatted_answer
                
            except Exception as e:
                logger.error(f"Error answering question {question_index}: {str(e)}")
                error_response = f"""Error processing question: {str(e)}
                The system encountered an issue while analyzing the document.
                This may be due to document processing limitations or connectivity issues.
                Please verify the document format and try again with a simpler question.
                Contact support if the problem persists with properly formatted documents."""
                return question_index, error_response
    
    async def answer_questions_parallel(self, gemini_file, questions: List[str]) -> List[str]:
        """Answer multiple questions about the document in parallel"""
        logger.info(f"Starting parallel processing of {len(questions)} questions...")
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
                    logger.info(f"Progress: {len(completed_tasks)}/{len(questions)} questions completed")
                except Exception as e:
                    logger.error(f"Task failed: {str(e)}")
                    error_response = f"""Task processing error: {str(e)}
                    The parallel processing encountered an unexpected issue.
                    This may be due to resource constraints or API limitations.
                    The system will attempt to process remaining questions.
                    Individual question results may vary in completeness."""
                    completed_tasks.append((len(completed_tasks) + 1, error_response))
        
        except Exception as e:
            logger.error(f"Error in parallel processing: {str(e)}")
            # If something goes wrong, fall back to sequential processing
            return await self.answer_questions_sequential(gemini_file, questions)
        
        # Sort answers by question index to maintain order
        completed_tasks.sort(key=lambda x: x[0])
        answers = [answer for _, answer in completed_tasks]
        
        processing_time = time.time() - start_time
        logger.info(f"Completed all {len(questions)} questions in {processing_time:.2f} seconds")
        logger.info(f"Average time per question: {processing_time/len(questions):.2f} seconds")
        
        return answers
    
    async def answer_questions_sequential(self, gemini_file, questions: List[str]) -> List[str]:
        """Fallback sequential processing method"""
        logger.info("Using sequential processing as fallback")
        answers = []
        
        for i, question in enumerate(questions, 1):
            try:
                logger.info(f"Processing question {i}/{len(questions)}: {question[:50]}...")
                
                # Use the enhanced prompt for sequential processing too
                prompt = self.create_enhanced_prompt(question)
                
                response = self.model.generate_content([prompt, gemini_file])
                answer = response.text.strip()
                
                # Ensure 5-line format
                lines = answer.split('\n')
                lines = [line.strip() for line in lines if line.strip()]
                
                if len(lines) != 5:
                    if len(lines) < 5:
                        while len(lines) < 5:
                            lines.append("Additional context may be limited in the source document.")
                    else:
                        lines = lines[:5]
                
                formatted_answer = '\n'.join(lines)
                answers.append(formatted_answer)
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error answering question {i}: {str(e)}")
                error_response = f"""Error processing question: {str(e)}
                Sequential processing encountered an issue with this specific question.
                This may be due to document complexity or question formatting.
                Please try rephrasing the question or check document accessibility.
                Other questions in the batch may process successfully."""
                answers.append(error_response)
        
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
        "message": "Enhanced Document Q&A API with Detailed Reasoning", 
        "version": "1.0.0",
        "features": ["enhanced_prompting", "detailed_responses", "5_line_format", "parallel_processing", "rate_limiting", "thread_safe"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "gemini_configured": bool(GEMINI_API_KEY),
        "max_concurrent_questions": processor.max_concurrent_questions,
        "response_format": "5_lines_detailed"
    }

@app.post("/api/v1/hackrx/run", response_model=DocumentResponse)
async def process_document(
    request: DocumentRequest,
    token: str = Depends(verify_token)
):
    """
    Process a document and answer questions about it using enhanced prompting for detailed responses
    
    - **documents**: URL to the document (PDF, DOCX, TXT)
    - **questions**: List of questions to answer about the document (each answer will be exactly 5 detailed lines)
    """
    
    try:
        logger.info(f"Processing document: {request.documents}")
        logger.info(f"Number of questions: {len(request.questions)}")
        
        # Download document
        temp_file_path = await processor.download_document(request.documents)
        
        # Upload to Gemini
        gemini_file = await processor.upload_to_gemini(temp_file_path)
        
        try:
            # Answer questions in parallel with enhanced prompting
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
    Process multiple documents in batch (each document's questions are processed in parallel)
    Each answer will be exactly 5 detailed lines with comprehensive reasoning
    """
    
    results = []
    
    for i, request in enumerate(requests, 1):
        try:
            logger.info(f"Processing batch item {i}/{len(requests)}")
            result = await process_document(request, token)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing batch item {i}: {str(e)}")
            error_response = f"""Batch processing error for document {i}: {str(e)}
            The system encountered an issue while processing this specific document.
            This may be due to document format, size, or accessibility issues.
            Other documents in the batch may process successfully.
            Please verify document format and accessibility for failed items."""
            results.append(DocumentResponse(answers=[error_response]))
    
    return results

@app.get("/api/v1/hackrx/status")
async def get_status(token: str = Depends(verify_token)):
    """Get API status and configuration"""
    return {
        "status": "operational",
        "gemini_model": "gemini-2.5-flash",
        "model_features": "enhanced_reasoning_capabilities",
        "max_file_size": "2GB",
        "supported_formats": ["PDF", "DOCX", "TXT"],
        "authentication": "Bearer token required",
        "parallel_processing": True,
        "max_concurrent_questions": processor.max_concurrent_questions,
        "response_format": {
            "lines": 5,
            "style": "detailed_comprehensive",
            "reasoning": "gemini_2.5_enhanced_analytical"
        },
        "features": {
            "enhanced_prompting": "Structured analytical framework",
            "detailed_responses": "5-line comprehensive answers",
            "contextual_reasoning": "Multi-step analysis process with Gemini 2.5 capabilities",
            "advanced_thinking": "Utilizing Gemini 2.5 Flash enhanced reasoning",
            "rate_limiting": "Built-in semaphore control",
            "thread_safety": "Thread-local model instances",
            "error_handling": "Graceful fallback to sequential processing",
            "progress_tracking": "Real-time completion logging"
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
        "response_format": "5_lines_detailed_maintained"
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
    
    logger.info(f"Starting enhanced document Q&A server on {HOST}:{PORT}")
    logger.info(f"Max concurrent questions: {processor.max_concurrent_questions}")
    logger.info(f"Response format: 5 detailed lines with comprehensive reasoning")
    logger.info(f"API Base URL: http://localhost:{PORT}/api/v1")
    logger.info(f"Documentation: http://localhost:{PORT}/docs")
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )
