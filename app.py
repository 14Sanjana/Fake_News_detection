from flask import Flask, render_template, request
import pickle, re, nltk, requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deep_translator import GoogleTranslator  # More reliable than googletrans

app = Flask(__name__)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load your trained model
model = load_model("model/cnn_bilstm_fakenews_model.h5")

# Load tokenizer
with open("tokenizer/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200

# -----------------------
# Helper functions
# -----------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join(w for w in text.split() if w not in stop_words)

def extract_from_url(url):
    try:
        page = requests.get(url, timeout=5)
        soup = BeautifulSoup(page.text, "html.parser")
        return " ".join(p.text for p in soup.find_all("p"))
    except:
        return ""

def translate_to_english(text):
    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        return translated_text
    except:
        return text  # fallback to original if translation fails

# -----------------------
# Routes
# -----------------------

@app.route("/")
def landing():
    return render_template("landing.html", page="home")

@app.route("/about")
def about():
    return render_template("about.html", page="about")

@app.route("/mode")
def mode():
    return render_template("mode.html", page="mode")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    mode = request.args.get("type")

    if request.method == "POST":
        user_input = request.form["news"]

        if mode == "url":
            text = extract_from_url(user_input)
        else:
            text = user_input

        
        try:
            translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        except:
            translated_text = text

        detected_lang = "en" if translated_text == text else "non-en"
        text_for_model = translated_text

        cleaned = clean_text(text_for_model)
        seq = tokenizer.texts_to_sequences([cleaned])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

        prob = model.predict(pad)[0][0]

        if prob >= 0.6:
            prediction = "REAL NEWS 🟢"
            confidence = round(prob * 100, 2)
        else:
            prediction = "FAKE NEWS 🔴"
            confidence = round((1 - prob) * 100, 2)

        return render_template(
            "result.html",
            prediction=prediction,
            confidence=confidence,
            detected_lang=detected_lang,
            translated_text=translated_text,
            page="result"
        )

    return render_template("predict.html", mode=mode, page="mode")

# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
