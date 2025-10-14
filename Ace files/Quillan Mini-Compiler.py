#!/usr/bin/env python3
# Quillan Code Executor - A multi-stage code analysis and execution tool.
# This script performs static code checks, compilation, and execution,
# providing comprehensive output for debugging.

import subprocess
import os
import sys
import shutil

def check_tool_exists(name):
    """Check if a command-line tool is installed and available in the system's PATH."""
    return shutil.which(name) is not None

def execute_stage(stage_name, command_list, file_path):
    """
    Executes a single stage of the code analysis pipeline and prints the results.
    
    Args:
        stage_name (str): The name of the stage (e.g., "Check", "Compile").
        command_list (list): The list of strings representing the command.
        file_path (str): The path to the file being processed.
        
    Returns:
        tuple: (return_code, stdout, stderr)
    """
    print(f"--- {stage_name} Stage ---")
    try:
        command = [cmd.replace("{file_path}", file_path) for cmd in command_list]
        result = subprocess.run(command, capture_output=True, text=True, errors='ignore')
        
        print(f"Command: {' '.join(command)}")
        if result.stdout:
            print("\n-- Standard Output --")
            print(result.stdout)
        if result.stderr:
            print("\n-- Standard Error --")
            print(result.stderr)
            
        return result.returncode, result.stdout, result.stderr
        
    except FileNotFoundError:
        print(f"Error: The required tool '{command_list[0]}' was not found.")
        return 1, "", ""
    except Exception as e:
        print(f"An unexpected error occurred during the {stage_name} stage: {e}")
        return 1, "", ""

def ace_execute_code(file_path):
    """
    Main execution pipeline for the Quillan Code Executor.
    
    Args:
        file_path (str): The path to the code file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    # A modular configuration for different languages.
    # Add new languages by creating a new dictionary entry.
    # The '{file_path}' placeholder will be replaced at runtime.
    LANG_CONFIG = {
        '.py': {
            'check': ['pylint', '{file_path}'],
            'run': ['python3', '{file_path}'],
            'description': 'Python (requires python3 and pylint)'
        },
        '.json': {
            'check': ['jsonlint', '{file_path}'],
            'description': 'JSON (requires jsonlint)'
        },
        '.yaml': {
            'check': ['yamllint', '{file_path}'],
            'description': 'YAML (requires yamllint)'
        },
        '.js': {
            'check': ['eslint', '{file_path}'],
            'run': ['node', '{file_path}'],
            'description': 'JavaScript (requires node and eslint)'
        },
        '.html': {
            'check': ['html-validate', '{file_path}'],
            'description': 'HTML (requires html-validate)'
        },
        '.css': {
            'check': ['stylelint', '{file_path}'],
            'description': 'CSS/Tailwind (requires stylelint)'
        },
        '.c': {
            'compile': ['gcc', '-o', 'a.out', '{file_path}'],
            'run': ['./a.out'],
            'description': 'C (requires gcc)'
        },
        '.cpp': {
            'compile': ['g++', '-o', 'a.out', '{file_path}'],
            'run': ['./a.out'],
            'description': 'C++ (requires g++)'
        },
    }

    # Get the file extension to determine the language
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension not in LANG_CONFIG:
        print(f"Error: Unsupported file extension '{file_extension}'")
        print("Supported extensions are: " + ", ".join(LANG_CONFIG.keys()))
        return
        
    config = LANG_CONFIG[file_extension]
    print(f"Processing '{file_path}' as {config['description']}...")

    try:
        # --- STAGE 1: CHECKING ---
        if 'check' in config:
            check_return_code, _, _ = execute_stage("Code Check", config['check'], file_path)
            if check_return_code != 0:
                print("Code check failed. Execution halted.")
                return

        # --- STAGE 2: COMPILATION ---
        if 'compile' in config:
            compile_return_code, _, _ = execute_stage("Compilation", config['compile'], file_path)
            if compile_return_code != 0:
                print("Compilation failed. Execution halted.")
                return
            
        # --- STAGE 3: EXECUTION ---
        if 'run' in config:
            run_return_code, _, _ = execute_stage("Execution", config['run'], file_path)
            if run_return_code != 0:
                print("Execution failed.")
                return
        
        # --- STAGE 4: FINAL SUCCESS ---
        print("\nAll applicable stages completed successfully.")

    finally:
        # --- STAGE 5: CLEANUP ---
        if 'compile' in config and os.path.exists('a.out'):
            os.remove('a.out')
            print("\nCleaned up compiled executable 'a.out'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ace_code_executor.py <path_to_file>")
        print("This script provides a universal foundation for your code execution needs.")
        sys.exit(1)
    
    ace_execute_code(sys.argv[1])
