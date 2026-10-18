import json

# Test the exact JSON structure from the file
with open('Quillan_finetune_full_dataset.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            print(f"Line {i} keys:", list(data.keys()))
            
            # Check Output_Sections for Final output
            if 'Output_Sections' in data:
                output_sections = data['Output_Sections']
                print("Output_Sections keys:", list(output_sections.keys()))
                
                # Look for content in various possible keys
                for key in ['Final output', 'Final output', 'Final output']:
                    if key in output_sections:
                        print(f"Found content in '{key}'")
                        content = output_sections[key]
                        print(f"Content length: {len(str(content))}")
                        print(f"Content preview: {str(content)[:200]}...")
                        break
                else:
                    print("Available Output_Sections keys:", list(output_sections.keys()))
            
            if i > 2:  # Only check first few lines
                break
        except Exception as e:
            print(f"Line {i} JSON error: {e}")
            continue
