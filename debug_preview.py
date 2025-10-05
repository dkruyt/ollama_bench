#!/usr/bin/env python3
"""Debug script to test preview functionality"""
import time
from ollama import Client

client = Client(host="http://localhost:11434")

# Test streaming with manual tracking
accumulated = ""
chunk_count = 0

print("Testing streaming...")
for chunk in client.generate(model="llama3.1:8b", prompt="Say hello in 5 words", stream=True):
    if hasattr(chunk, 'response'):
        resp_text = chunk.response or ""
        if resp_text:
            accumulated += resp_text
            chunk_count += 1
            print(f"Chunk {chunk_count}: '{resp_text}' | Accumulated: '{accumulated}'")

print(f"\nFinal accumulated: '{accumulated}'")
print(f"Total chunks: {chunk_count}")
