import os
import sys

project_dir = os.path.join(os.path.dirname(__file__), "drone_detection")
sys.path.insert(0, project_dir)

main_file = os.path.join(project_dir, "main.py")
with open(main_file, encoding="utf-8") as file_handle:
    code = file_handle.read()
    exec(compile(code, main_file, "exec"), globals())
