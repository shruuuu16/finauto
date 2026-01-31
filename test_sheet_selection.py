
import requests
import os
import time

URL = "http://127.0.0.1:8000"
INPUT_DIR = "input"
# Pick an existing PDF from input dir listing
PDF_PATH = os.path.join(INPUT_DIR, "073b136d-d6e4-4b05-8fc9-4b6f10b25d5b_doc.pdf")
# Use the user-provided template path
TEMPLATE_PATH = r"c:\Users\shrey\OneDrive\Desktop\finauto\GSTR1_1.1.xlsx"

def test_manual_sheet():
    print(f"Testing manual sheet selection with 'B2CS'...")
    
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF not found at {PDF_PATH}")
        return

    if not os.path.exists(TEMPLATE_PATH):
        print(f"Error: Template not found at {TEMPLATE_PATH}")
        return

    files = {
        'document': open(PDF_PATH, 'rb'),
        'template': open(TEMPLATE_PATH, 'rb')
    }
    data = {
        'sheet_name': 'B2CS'
    }

    try:
        print("Sending upload request...")
        response = requests.post(f"{URL}/upload", files=files, data=data, timeout=10)
        print(f"Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"Upload failed: {response.text}")
            return
        
        job_id = response.json()['job_id']
        print(f"Job started: {job_id}")
        
        # Poll status
        for _ in range(20): # Wait up to 20 seconds
            time.sleep(1)
            status_resp = requests.get(f"{URL}/status/{job_id}")
            if status_resp.status_code == 200:
                job_info = status_resp.json()
                status = job_info.get('status')
                if status == 'done':
                    print("Job completed.")
                    # Check selected sheet
                    selected = job_info.get('selected_sheet')
                    print(f"Selected Sheet (Result): {selected}")
                    
                    if selected == 'B2CS':
                        print("SUCCESS: Manual sheet selection worked.")
                    else:
                        print(f"FAILURE: Expected 'B2CS', got '{selected}'")
                        
                    # Also check message
                    print(f"Message: {job_info.get('message')}")
                    return
                elif status == 'failed':
                    print(f"Job failed: {job_info.get('error')}")
                    return
        
        print("Timeout waiting for job completion.")
        
    except Exception as e:
        print(f"Test failed with exception: {e}")

if __name__ == "__main__":
    test_manual_sheet()
