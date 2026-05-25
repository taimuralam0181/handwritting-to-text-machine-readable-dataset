# list_my_models.py
import google.generativeai as genai

# Your API key
API_KEY = ""  # Replace with your actual key

genai.configure(api_key=API_KEY)

print("=" * 60)
print("Models AVAILABLE to YOUR API key:")
print("=" * 60)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"  ✅ {model.name}")