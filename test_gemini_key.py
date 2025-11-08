from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env file from the current directory
# Try multiple paths
env_paths = [
    Path(__file__).parent / '.env',
    Path('.env'),
    Path.cwd() / '.env'
]

loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        loaded = True
        print(f"Loaded .env from: {env_path}")
        break

if not loaded:
    # Try loading without path (default behavior)
    load_dotenv(override=True)

# Debug: Print all env vars starting with GEMINI
print("\nDebug - Environment variables:")
for key, value in os.environ.items():
    if 'GEMINI' in key.upper():
        print(f"  {key} = {value[:20]}...")

# Check if API key is loaded
api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key in ["your_api_key_here", "your_gemini_api_key_here"]:
    print("ERROR: GEMINI_API_KEY not set or using placeholder value")
    print(f"   Current value: {api_key if api_key else 'None'}")
    exit(1)

print(f"[OK] API Key found: {api_key[:10]}...{api_key[-4:]}")
print("\nTesting Gemini API key...\n")

# Test 1: Embedder
print("1. Testing Gemini Embedder...")
try:
    from rag.embedder import GeminiEmbedder
    emb = GeminiEmbedder()
    vec = emb.embed_text("test")
    print(f"   [OK] Embedder working - Generated embedding vector of length {len(vec)}")
except Exception as e:
    print(f"   [ERROR] Embedder failed: {str(e)}")
    exit(1)

# Test 2: Summarizer
print("2. Testing Gemini Summarizer...")
try:
    from rag.summarizer import GeminiSummarizer
    summarizer = GeminiSummarizer()
    response = summarizer.model.generate_content("Say 'Hello' if you can read this.")
    print(f"   [OK] Summarizer working - Response: {response.text[:50]}...")
except Exception as e:
    print(f"   [ERROR] Summarizer failed: {str(e)}")
    exit(1)

print("\n[SUCCESS] All tests passed! Gemini API key is working correctly.")
