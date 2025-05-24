from google import genai 
from google.genai import types  
import os 
import subprocess  
import argparse  
from prompts import PROMPT  
from datetime import datetime  
import concurrent.futures  
import hashlib  

client = genai.Client(api_key="")  # Initializes a client for the Google generative AI model (API key needs to be filled).

def get_input(json_path, file_index, save_path):  # Creates a prompt for the AI based on a JSON file.
    with open(json_path, "r", encoding="utf-8") as f:
        json_str = f.read()
    return f"'''Give me CAD query from this CAD sequence: {json_str}. The export file name should be {os.path.join(save_path, file_index)}.stl. In the end, only save stl file, don't need to use show().'''", json_str

def get_pure_python(gemini_output):  # Cleans up AI-generated code by removing Markdown markers.
    if gemini_output.startswith("```python"):
        gemini_output = gemini_output[len("```python"):].strip()
    if gemini_output.endswith("```"):
        gemini_output = gemini_output[:-len("```")].strip()
    return gemini_output

def generate_code_with_timeout(chat, user_input, timeout=60):  # Generates code with a timeout using threading.
    def send_message_wrapper():  # Helper function to run the AI request.
        response = chat.send_message(user_input)
        return get_pure_python(response.text)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(send_message_wrapper)  # Submit the AI request to a thread.
        try:
            result = future.result(timeout=timeout)  # Wait for result with timeout.
            return result
        except concurrent.futures.TimeoutError:
            print(f"Generation timed out after {timeout} seconds")
            return None
        except Exception as e:
            print(f"Error generating code: {e}")
            return None

def generate_and_validate_code(chat, json_str, file_index, cq_dir, stl_dir, max_attempts=5, timeout=60):  # Generates and validates CAD code with iterative retries.
    response = generate_code_with_timeout(chat, json_str, timeout)
    if response is None:
        return False, os.path.join(cq_dir, f"{file_index}.py"), ["Initial generation failed (timeout or error)"], response
    
    cq_path = os.path.join(cq_dir, f"{file_index}.py")
    stl_path = os.path.join(stl_dir, f"{file_index}.stl")
    attempt = 0
    error_messages = []
    previous_error_hash = None
    same_error_count = 0

    while attempt < max_attempts:
        attempt += 1
        print(f"Attempt {attempt} for {file_index}")
        with open(cq_path, "w", encoding="utf-8") as f:
            f.write(response)

        result = subprocess.run(['python', cq_path], capture_output=True, text=True)
        # Check if STL file exists as a fallback for success
        stl_exists = os.path.exists(stl_path)
        print(f"STL file exists: {stl_exists}, Return code: {result.returncode}")
        if result.returncode == 0 or stl_exists:
            print(f"Success for {file_index} after {attempt} attempt(s)")
            return True, cq_path, error_messages, response
        
        error_msg = result.stderr.strip() or result.stdout.strip() or "No output captured"
        error_messages.append(f"Attempt {attempt} error: {error_msg}, Return code: {result.returncode}, Stdout: {result.stdout.strip()}")
        print(f"{file_index} meets error on attempt {attempt}: {error_msg[:100]}...")
        print(f"Stdout: {result.stdout.strip()[:100]}...")

        # Check if the same error occurred
        error_hash = hashlib.md5(error_msg.encode('utf-8')).hexdigest()
        if error_hash == previous_error_hash:
            same_error_count += 1
        else:
            same_error_count = 1
        previous_error_hash = error_hash

        if same_error_count >= 2:
            print(f"Same error occurred {same_error_count} times for {file_index}, stopping retries")
            error_messages.append(f"Stopped after {attempt} attempts due to repeated error")
            return False, cq_path, error_messages, response

        if attempt < max_attempts:
            retry_prompt = (
                f"Code attempt {attempt} for CAD sequence failed with error: {error_msg}\n"
                f"Stdout: {result.stdout.strip()}\n"
                f"Original code:\n{response}\n"
                f"Generate corrected Python code for the CAD sequence. Ensure the code is valid, avoids the previous error, "
                f"and correctly exports an STL file to {os.path.join(stl_dir, f'{file_index}.stl')}. "
                f"Use proper CAD library syntax (e.g., CadQuery: `from cadquery import Workplane`). "
                f"Do not use show(). Only provide the Python code."
            )
            response = generate_code_with_timeout(chat, retry_prompt, timeout)
            if response is None:
                error_messages.append(f"Attempt {attempt + 1} generation failed (timeout or error)")
                return False, cq_path, error_messages, response
        else:
            print(f"Max attempts ({max_attempts}) reached for {file_index}")
    
    return False, cq_path, error_messages, response

def main():  # Orchestrates the script’s logic.
    success_count = 0
    fail_count = 0
    failed_scripts = []

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root_dir', type=str, required=True, help="Directory containing input JSON files")
    parser.add_argument('-s', '--stl_dir', type=str, required=True, help="Directory to save STL files")
    parser.add_argument('-c', '--cq_dir', type=str, required=True, help="Directory to save generated Python scripts")
    parser.add_argument('--max_attempts', type=int, default=5, help="Maximum number of retry attempts for code generation")
    parser.add_argument('--timeout', type=int, default=60, help="Timeout in seconds for AI requests")
    args = parser.parse_args()
    root_dir = args.root_dir
    stl_dir = args.stl_dir
    cq_dir = args.cq_dir
    max_attempts = args.max_attempts
    timeout = args.timeout
    
    os.makedirs("./failed_scripts", exist_ok=True)
    
    for file in os.listdir(root_dir):
        if file.endswith('json'):
            print(f'Working with {file}')
            file_path = os.path.join(root_dir, file)
            file_index = file.split('_')[0]
            json_str, raw_json = get_input(file_path, file_index, stl_dir)
            chat = client.chats.create(model="gemini-2.0-flash", config=types.GenerateContentConfig(
                temperature=0, system_instruction=PROMPT
            ))

            success, path, errors, last_code = generate_and_validate_code(chat, json_str, file_index, cq_dir, stl_dir, max_attempts, timeout)
            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_scripts.append((path, errors, last_code, raw_json))

    print(f"Number of successes: {success_count}")
    print(f"Number of failures: {fail_count}")
    with open(os.path.join("./failed_scripts/", f'{os.path.basename(cq_dir)}_failed_scripts.txt'), 'w') as f:
        for path, errors, code, json_content in failed_scripts:
            f.write(f"Failed script: {path}\n")
            f.write(f"Input JSON: {json_content}\n")
            f.write(f"Last generated code:\n{code if code else 'None (generation failed)'}\n")
            f.write("Errors:\n")
            for error in errors:
                f.write(f"{error}\n")
            f.write("\n")

if __name__ == "__main__":
    start = datetime.now()
    main()
    end = datetime.now()
    print(f"Start: {start.strftime('%H:%M:%S')}")
    print(f"End:   {end.strftime('%H:%M:%S')}")
    print(f"Total execution time: {end - start}")
