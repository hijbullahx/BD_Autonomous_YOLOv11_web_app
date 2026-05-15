import streamlit as st
import sys
import os

# Add the breast_cancer_detection directory to the path
project_dir = os.path.join(os.path.dirname(__file__), "breast_cancer_detection")
sys.path.insert(0, project_dir)

# Now execute the main project file
main_file = os.path.join(project_dir, "main.py")
with open(main_file, encoding='utf-8') as f:
    code = f.read()
    globals()["__file__"] = main_file
    exec(compile(code, main_file, "exec"), globals())
