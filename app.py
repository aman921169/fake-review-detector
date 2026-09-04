# app.py
# Gradio demo for the fake review detector (broad multi-category dataset)
# Loads saved models instantly instead of retraining

import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import gradio as gr

print("Loading saved models...")

# ---------- Load baseline ----------
baseline_model = joblib.load("baseline_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# ---------- Load BERT ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained("./bert_broad_model")
bert_model = DistilBertForSequenceClassification.from_pretrained("./bert_broad_model")
bert_model = bert_model.to(device)
bert_model.eval()

print("Models loaded! Launching app...")

# ---------- Prediction function ----------
def predict_review(review_text):
    if not review_text.strip():
        return "Please paste a review first.", "", ""

    # Baseline prediction
    review_tfidf = vectorizer.transform([review_text])
    baseline_pred = baseline_model.predict(review_tfidf)[0]
    baseline_proba = baseline_model.predict_proba(review_tfidf)[0]
    baseline_confidence = max(baseline_proba) * 100
    baseline_label = "FAKE (AI-generated)" if baseline_pred == "CG" else "REAL (human-written)"

    # BERT prediction
    inputs = tokenizer(review_text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    bert_pred_idx = torch.argmax(probs).item()
    bert_label = "FAKE (AI-generated)" if bert_pred_idx == 1 else "REAL (human-written)"
    bert_confidence = probs[bert_pred_idx].item() * 100

    baseline_result = f"{baseline_label} ({baseline_confidence:.1f}% confident)"
    bert_result = f"{bert_label} ({bert_confidence:.1f}% confident)"
    agreement = "✅ Models agree" if baseline_label == bert_label else "⚠️ Models disagree"

    return baseline_result, bert_result, agreement

# ---------- Build the interface ----------
demo = gr.Interface(
    fn=predict_review,
    inputs=gr.Textbox(lines=6, placeholder="Paste a product review here...", label="Review Text"),
    outputs=[
        gr.Textbox(label="Baseline Model (TF-IDF + Logistic Regression)"),
        gr.Textbox(label="BERT Model (Fine-tuned DistilBERT)"),
        gr.Textbox(label="Agreement"),
    ],
    title="Fake Review Detector",
    description="Paste a product review (electronics, books, home goods, etc.) and see whether two different models think it's real or AI-generated.",
)

demo.launch()