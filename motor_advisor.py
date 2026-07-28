"""
Motor Maintenance Advisor - Google Gemini AI
Reads ml_model.json + motor image, gives maintenance suggestions

Setup:
  1. Get free API key: https://aistudio.google.com/apikey
  2. Install: pip install google-genai pillow
  3. Edit API_KEY below, then run: python motor_advisor.py "image.jpg"
"""

from google import genai
import json
import sys
import os
from PIL import Image

API_KEY = "AQ.Ab8RN6KuiWoOPSh-rAYlFZSyPKX5dRxfpU6eRUq7Bi6ZY0KDLg"
client = genai.Client(api_key=API_KEY)

PROMPT = """You are a motor maintenance expert. Analyze the motor image and prediction data below.
Provide:
1. What this motor condition/fault means in simple terms
2. Urgency level (Low / Medium / High / Critical)
3. Immediate actions to take
4. Recommended maintenance schedule
5. Preventive measures for the future

Prediction Data:
{json_data}"""

def main():
    if not os.path.exists("ml_model.json"):
        print("Run predict_cli.py first to generate ml_model.json")
        return
    
    with open("ml_model.json") as f:
        prediction = json.load(f)
    
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    parts = [PROMPT.format(json_data=json.dumps(prediction, indent=2))]
    
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path)
        parts.append(img)
        print(f"Analyzing motor image: {image_path}")
    elif image_path:
        print(f"Image not found: {image_path}")
    else:
        print("No image provided - text only analysis")
    
    print("Getting AI recommendations...")
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=parts
        )
        print("\n" + "=" * 55)
        print("MOTOR MAINTENANCE ADVISOR REPORT")
        print("=" * 55)
        print(response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
