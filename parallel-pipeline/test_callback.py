#!/usr/bin/env python3
import sys
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

def process_frame(frame):
    print("CALLBACK EXECUTED!")
    return True
