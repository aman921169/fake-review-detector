Fake Review Detector

A machine learning project that detects fake (AI-generated) product reviews, comparing a classic TF-IDF + Logistic Regression baseline against a fine-tuned DistilBERT model.

Includes a live demo (Gradio) where you can paste any review and see predictions from both models side by side.

Overview

Online reviews shape purchasing decisions, but a growing share of them are AI-generated rather than written by real customers. This project builds and compares two approaches to detecting this:

Baseline: TF-IDF vectorization + Logistic Regression
Fine-tuned transformer: DistilBERT, fine-tuned on labeled review data

The goal was to understand not just which model performs better, but why, and how much a transformer's contextual understanding actually buys you over a simple word-frequency approach.

Dataset

Fake Reviews Dataset (Kaggle) — 40,432 product reviews across 10 categories (Electronics, Books, Home & Kitchen, Toys, Pet Supplies, and more), labeled as either:

OR — Original, human-written review
CG — Computer-generated (AI-written) review

The dataset is perfectly balanced: 20,216 reviews of each class.

Results
Model	Accuracy	Precision (avg)	Recall (avg)
TF-IDF + Logistic Regression	86.2%	0.86	0.86
Fine-tuned DistilBERT	98.3%	0.98	0.98

Fine-tuning DistilBERT improved accuracy by ~12 percentage points over the baseline — a much larger gap than is typical for tasks like sentiment analysis. This makes sense given what's being detected: AI-generated text has structural and contextual patterns (generic phrasing, unnaturally consistent structure) that a model with real language understanding can pick up on far better than a word-frequency count ever could.