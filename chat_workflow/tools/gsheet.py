import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Callable
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# Load environment variables
load_dotenv()

# Define Google Sheets API scope
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly', 'https://www.googleapis.com/auth/drive.readonly']

def authenticate_google() -> Credentials:
    """
    Authenticate with Google APIs using OAuth 2.0.
    """
    creds = None
    # Load credentials from file if available
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If no valid credentials, prompt user to log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save credentials for future use
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def get_all_sheets(folder_link: str) -> List[Dict[str, Any]]:
    """
    Retrieve all OKR data from Google Sheets in a specified folder.

    Args:
        folder_link: Link to the Google Drive folder containing OKR sheets.

    Returns:
        List of dictionaries containing OKR data from all sheets.
    """
    # Authenticate with Google APIs
    creds = authenticate_google()
    drive_service = build('drive', 'v3', credentials=creds)
    sheets_service = build('sheets', 'v4', credentials=creds)

    # Extract folder ID from the folder link
    folder_id = folder_link.split('/')[-1]

    # List all files in the folder
    query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.spreadsheet'"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])

    # Retrieve data from each sheet
    okr_data = []
    for file in files:
        sheet_id = file['id']
        sheet_name = file['name']
        # Get sheet data
        sheet = sheets_service.spreadsheets().get(spreadsheetId=sheet_id).execute()
        sheets = sheet.get('sheets', [])
        for sheet in sheets:
            sheet_title = sheet['properties']['title']
            range_name = f"{sheet_title}!A1:F100"  # Adjust range as needed
            result = sheets_service.spreadsheets().values().get(
                spreadsheetId=sheet_id, range=range_name
            ).execute()
            values = result.get('values', [])
            if values:
                okr_data.append({
                    "sheet_name": sheet_name,
                    "sheet_title": sheet_title,
                    "data": values
                })
    return okr_data

def is_gsheet_tool_available() -> bool:
    """
    Check if the Google Sheets tool is available.
    """
    # Check if credentials file exists
    return os.path.exists('credentials.json')

# A function that returns the Google Sheets tool if available
def get_gsheet_tool() -> List[Callable]:
    if is_gsheet_tool_available():
        return [get_all_sheets]
    else:
        return []