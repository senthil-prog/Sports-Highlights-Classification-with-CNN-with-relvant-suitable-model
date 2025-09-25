# Name That Sport! 

Using a deep CNN to classify images of sports

This project was a ton of fun for me! I wanted to learn how to build a CNN to classify images. As a massive sports fan, I felt there were no better images to classify than images of different sports. I had a few challenges while building this, but overall, it was a great learning experience for me.

## Installation and Running the Application

1. First, create and activate a Python virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run main.py
```

4. Open your web browser and go to http://localhost:8501 (if it doesn't open automatically)

5. Upload an image of a sport using the interface, and the model will predict which sport it shows!

Note: The model has about 84% accuracy and works best with smaller images. The model can classify 94 different sports.


