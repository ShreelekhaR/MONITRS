"""
Test which Gemini models are available on your project.

Usage:
    export GCP_PROJECT_ID=your-project-id
    python test_models.py
"""

import os
from google import genai

PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'ai-sandbox-dev-f139')

MODELS = [
    'gemini-2.5-flash-lite',
    'gemini-2.5-flash',
    'gemini-2.5-pro',
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-3.1-flash-lite',
    'gemini-3.1-flash',
    'gemini-3.1-pro',
]

LOCATIONS = ['global', 'us-central1']

for location in LOCATIONS:
    print(f"\nLocation: {location}")
    print("-" * 50)
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=location)
    for model in MODELS:
        try:
            r = client.models.generate_content(model=model, contents='Say OK')
            print(f"  {model:<30} ✓ OK")
        except Exception as e:
            err = str(e)[:60]
            print(f"  {model:<30} ✗ {err}")
