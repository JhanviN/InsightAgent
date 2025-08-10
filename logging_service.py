# import json
# import time
# from datetime import datetime
# from typing import List, Optional, Dict, Any
# import asyncio
# from concurrent.futures import ThreadPoolExecutor
# import gspread
# from google.oauth2.service_account import Credentials
# import os
# from dotenv import load_dotenv

# # load_dotenv()a

# class RequestLogger:
#     def __init__(self):
#         self.sheet = None
#         self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="Logger")
#         self._setup_google_sheets()
    
#     def _setup_google_sheets(self):
#         """Initialize Google Sheets connection"""
#         try:
#             # Load service account credentials from environment variable
#             creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
#             if not creds_json:
#                 print("⚠️ Google Sheets logging disabled - no credentials found")
#                 return
            
#             # Parse the JSON credentials
#             creds_dict = json.loads(creds_json)
            
#             # Define the scope
#             scope = [
#                 "https://spreadsheets.google.com/feeds",
#                 "https://www.googleapis.com/auth/drive"
#             ]
            
#             # Create credentials and authorize
#             creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
#             gc = gspread.authorize(creds)
            
#             # Get the spreadsheet (you'll need to create this and share it with the service account)
#             sheet_id = os.getenv("GOOGLE_SHEET_ID")
#             if not sheet_id:
#                 print("⚠️ Google Sheets logging disabled - no sheet ID found")
#                 return
                
#             spreadsheet = gc.open_by_key(sheet_id)
#             self.sheet = spreadsheet.sheet1  # Use the first sheet
            
#             # Initialize headers if the sheet is empty
#             try:
#                 if not self.sheet.row_values(1):  # Check if first row is empty
#                     headers = [
#                         "Timestamp", "Request_ID", "Document_URL", 
#                         "Questions_Count", "Questions", "Answers", 
#                         "Processing_Time", "Document_Processing_Time", 
#                         "Query_Processing_Time", "Success", "Error_Message"
#                     ]
#                     self.sheet.insert_row(headers, 1)
#                     print("✅ Google Sheets initialized with headers")
#             except Exception as e:
#                 print(f"⚠️ Could not initialize headers: {e}")
            
#             print("✅ Google Sheets logging enabled")
            
#         except Exception as e:
#             print(f"❌ Failed to setup Google Sheets: {e}")
#             self.sheet = None
    
#     def _prepare_log_data(self, request_data: Dict[str, Any]) -> List[str]:
#         """Prepare data for logging to Google Sheets"""
#         timestamp = datetime.now().isoformat()
        
#         # Extract data with safe defaults
#         document_url = request_data.get('document_url', '')
#         questions = request_data.get('questions', [])
#         answers = request_data.get('answers', [])
        
#         questions_str = json.dumps(questions) if questions else ''
#         answers_str = json.dumps(answers) if answers else ''
        
#         # Truncate long strings to avoid cell limits
#         questions_str = questions_str[:30000] if len(questions_str) > 30000 else questions_str
#         answers_str = answers_str[:30000] if len(answers_str) > 30000 else answers_str
        
#         row_data = [
#             timestamp,
#             request_data.get('request_id', ''),
#             document_url,
#             len(questions),
#             questions_str,
#             answers_str,
#             request_data.get('total_processing_time', 0),
#             request_data.get('document_processing_time', 0),
#             request_data.get('query_processing_time', 0),
#             request_data.get('success', True),
#             request_data.get('error_message', '')
#         ]
        
#         return row_data
    
#     def _log_to_sheets_sync(self, request_data: Dict[str, Any]):
#         """Synchronous method to log to Google Sheets"""
#         if not self.sheet:
#             return
            
#         try:
#             row_data = self._prepare_log_data(request_data)
#             self.sheet.insert_row(row_data, 2)  # Insert at row 2 (after headers)
#             print(f"✅ Logged request {request_data.get('request_id', 'unknown')} to Google Sheets")
            
#         except Exception as e:
#             print(f"❌ Failed to log to Google Sheets: {e}")
    
#     async def log_request(self, request_data: Dict[str, Any]):
#         """Async method to log request data"""
#         if not self.sheet:
#             return
            
#         # Run the logging in a thread to avoid blocking
#         loop = asyncio.get_event_loop()
#         try:
#             await loop.run_in_executor(
#                 self.executor, 
#                 self._log_to_sheets_sync, 
#                 request_data
#             )
#         except Exception as e:
#             print(f"❌ Async logging failed: {e}")
    
#     def log_request_sync(self, request_data: Dict[str, Any]):
#         """Synchronous method for non-async contexts"""
#         self._log_to_sheets_sync(request_data)
    
#     def close(self):
#         """Clean up resources"""
#         if self.executor:
#             self.executor.shutdown(wait=True)

# # Global logger instance
# request_logger = RequestLogger()

# # Helper function to generate request IDs
# def generate_request_id() -> str:
#     """Generate a unique request ID"""
#     import uuid
#     return str(uuid.uuid4())[:8]

# # Helper function to log API requests
# async def log_api_request(
#     request_id: str,
#     document_url: str,
#     questions: List[str],
#     answers: List[str],
#     total_time: float,
#     doc_time: float,
#     query_time: float,
#     success: bool = True,
#     error_message: str = ""
# ):
#     """Log API request with all relevant data"""
#     request_data = {
#         'request_id': request_id,
#         'document_url': document_url,
#         'questions': questions,
#         'answers': answers,
#         'total_processing_time': round(total_time, 2),
#         'document_processing_time': round(doc_time, 2),
#         'query_processing_time': round(query_time, 2),
#         'success': success,
#         'error_message': error_message
#     }
    
#     await request_logger.log_request(request_data)