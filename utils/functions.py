import re
import json

def extract_json(input_string):
    # Define regex patterns for the three cases
    patterns = [
        r"```json\s*({.*?})\s*```",       # Case 1: ```json { ... } ```
        r"```\s*({.*?})\s*```",           # Case 2: ``` { ... } ```
        r"({.*?})"                        # Case 3: { ... }
    ]
    
    for pattern in patterns:
        match = re.search(pattern, input_string, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)  # Parse the matched JSON string
            except json.JSONDecodeError:
                pass  # Continue if parsing fails
    
    raise ValueError("No valid JSON object found in the input string")

def normalize_string(input_string):
    input_string = re.sub(r'\W+', '_', input_string)
    return input_string.lower()