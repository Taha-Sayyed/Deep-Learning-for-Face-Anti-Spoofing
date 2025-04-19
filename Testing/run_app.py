#!/usr/bin/env python
import os
import subprocess
import sys

def main():
    """Run the Streamlit application"""
    # Check if Streamlit is installed
    try:
        import streamlit
        print("Streamlit is installed.")
    except ImportError:
        print("Streamlit is not installed. Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Run the Streamlit app
    print("Starting Face Anti-Spoofing Detection App...")
    subprocess.call(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    main() 