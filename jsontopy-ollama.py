import requests
import os
import subprocess
import argparse
from datetime import datetime
import time
import threading
import json
from prompts import PROMPT



# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3"

def get_input(json_path, file_index, save_path):
    with open(json_path, "r", encoding="utf-8") as f:
        json_str = f.read()
    return f"'''Give me CAD query from this CAD sequence: {json_str}. The export file name should be {os.path.join(save_path, file_index)}.stl. In the end, only save stl file, don't need to use show().'''"

def get_pure_python(ollama_output):
    if ollama_output.startswith("```python"):
        ollama_output = ollama_output[len("```python"):].strip()
    if ollama_output.endswith("```"):
        ollama_output = ollama_output[:-len("```")].strip()
    return ollama_output

def generate_code_with_timeout(user_input, timeout=60):
    # Event to signal thread completion
    result_event = threading.Event()
    result_container = [None]  # To store the result or exception

    def api_request():
        try:
            # Prepare the payload for Ollama API
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": f"{PROMPT}\n\n{user_input}",
                "stream": False,
                "options": {
                    "temperature": 0
                }
            }
            # Make request to Ollama API
            response = requests.post(OLLAMA_API_URL, json=payload)
            response.raise_for_status()  # Raise exception for bad status codes
            result = response.json().get("response", "")
            result_container[0] = get_pure_python(result)
        except requests.RequestException as e:
            result_container[0] = e
        finally:
            result_event.set()  # Signal that the request is complete

    # Start the API request in a separate thread
    thread = threading.Thread(target=api_request)
    thread.start()

    # Wait for the thread to complete or timeout
    if not result_event.wait(timeout):
        print(f"Generation timed out after {timeout} seconds")
        return None

    # Check the result
    if isinstance(result_container[0], Exception):
        print(f"Error generating code: {result_container[0]}")
        return None
    return result_container[0]

def generate_and_validate_code(json_str, file_index, cq_dir):
    response = generate_code_with_timeout(json_str)
    if response is None:
        return False, os.path.join(cq_dir, f"{file_index}.py")
    cq_path = os.path.join(cq_dir, f"{file_index}.py")

    for attempt in range(2):
        with open(cq_path, "w", encoding="utf-8") as f:
            f.write(response)

        result = subprocess.run(['python', cq_path], capture_output=True, text=True)
        if result.returncode == 0:
            return True, cq_path
        else:
            if attempt == 0:
                print(f"{file_index} meets error")
                error_msg = '\n'.join(result.stderr.splitlines()[-5:])
                retry_prompt = f'code: {response} has an error: {error_msg}, generate it again, only give me python code'
                response = generate_code_with_timeout(retry_prompt)
                if response is None:
                    return False, cq_path
    return False, cq_path

def main():
    success_count = 0
    fail_count = 0
    failed_scripts = []

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root_dir', type=str, required=True)
    parser.add_argument('-s', '--stl_dir', type=str, required=True)
    parser.add_argument('-c', '--cq_dir', type=str, required=True)
    args = parser.parse_args()
    root_dir = args.root_dir
    stl_dir = args.stl_dir
    cq_dir = args.cq_dir
    
    os.makedirs("./failed_scripts", exist_ok=True)
    
    for file in os.listdir(root_dir):
        if file.endswith('json'):
            print(f'working with {file}')
            file_path = os.path.join(root_dir, file)
            file_index = file.split('_')[0]
            json_str = get_input(file_path, file_index, stl_dir)
            
            success, path = generate_and_validate_code(json_str, file_index, cq_dir)
            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_scripts.append(path)

    print(f"Number of success: {success_count}")
    print(f"Number of failure: {fail_count}")
    with open(os.path.join("./failed_scripts/", f'{os.path.basename(cq_dir)}_failed_scripts.txt'), 'w') as f:
        for item in failed_scripts:
            f.write(str(item) + '\n')

if __name__ == "__main__":
    start = datetime.now()
    main()
    end = datetime.now()
    print(f"Start: {start.strftime('%H:%M:%S')}")
    print(f"End:   {end.strftime('%H:%M:%S')}")
    print(f"Total execution time: {end - start}")