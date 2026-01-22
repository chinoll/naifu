"""
Upload Patch Script for SDXL Training Server

Reads a Python file and sends it to the server for remote execution (hot-patching).

Usage:
    python training_diffusers/upload_patch.py my_patch.py --server http://localhost:8000 --password mypassword
"""

import argparse
import os
import sys
import json
import urllib.request
import urllib.error

def main():
    parser = argparse.ArgumentParser(description="Upload Python code patch to SDXL Server")
    parser.add_argument("file", type=str, help="Path to Python file containing the patch code")
    parser.add_argument("--server", type=str, default="http://localhost:8000", help="Server URL")
    parser.add_argument("--password", type=str, default=None, help="Server password (optional)")
    
    args = parser.parse_args()
    
    # Resolve password
    password = args.password or os.environ.get("SDXL_SERVER_PASSWORD")
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        sys.exit(1)
        
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            code_content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
        
    payload = {
        "code": code_content
    }
    if password:
        payload["password"] = password
        
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        f"{args.server.rstrip('/')}/code/execute",
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Uploading patch from {args.file} to {args.server}...")
    
    try:
        with urllib.request.urlopen(req) as response:
            resp_body = response.read().decode('utf-8')
            print(f"Server Response: {resp_body}")
            
            if response.status == 200:
                print("Patch executed successfully.")
            else:
                print("Warning: Non-200 status code returned.")
                
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        try:
            print(e.read().decode('utf-8'))
        except:
            pass
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection Error: {e.reason}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
