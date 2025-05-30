# Bilingual NLP: Urdu-to-English Neural Machine Translation

This project presents a Neural Machine Translation (NMT) model designed to convert informal, incorrect, or noisy Urdu sentences into fluent English translations. Built using HuggingFace Transformers and Helsinki-NLP’s OPUS-MT architecture, the model was trained on a dataset of over 131,000 Urdu-English sentence pairs and achieved a BLEU score of 48%.

---

## Project Summary

- **Objective:** Translate incorrect Urdu text into grammatically correct English.
- **Model:** Neural Machine Translation using Transformer architecture.
- **Dataset Size:** 131,678 Urdu-English pairs.
- **Train-Test Split:** 80% training / 20% testing.
- **Evaluation Metric:** BLEU Score — **48%**.
- **Libraries Used:** HuggingFace, Helsinki-NLP (OPUS-MT), Scikit-learn, Pandas.

---

## Core Technologies

- HuggingFace Transformers  
- Helsinki-NLP (OPUS-MT models)  
- Python (Pandas, NumPy, Sklearn)  
- BLEU Score Evaluation  
- Google Colab (Model Training & Testing)  
- Gradio UI (for Sentiment Analysis module)

---

## File Structure

📦 bilingual-nlp-urdu-english
┣ 📄 urdu_to_english_nmt.ipynb
┣ 📄 urdu_sentiment_gradio.py
┣ 📄 All_Drama_Sentiment_Analysis_UPDATED.xlsx
┣ 📄 Positive_Urdu_Words_Corpus.txt
┣ 📄 Negative_Urdu_Words_Corpus.txt
┣ 📄 requirements.txt
┣ 📄 README.md
┗ 📁 screenshots/
┣ Screenshot (361).png
┣ Screenshot (362).png
┣ Screenshot (363).png
┗ Screenshot (364).png


---

## Google Colab

> You can run and test the NMT model directly in Colab.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x3P-UoLPe9X9Ym7w0wGYf-1sjqmfO_B1?usp=sharing)

---

## Screenshots

| NMT Training Output | BLEU Score Evaluation |
|---------------------|------------------------|
| ![Training Output](Screenshot%20(361).png) | ![BLEU Score](Screenshot%20(362).png) |

| Sentiment Model UI | Urdu-to-English Translation |
|--------------------|-----------------------------|
| ![Gradio UI](Screenshot%20(363).png) | ![Translation Example](Screenshot%20(364).png) |

---

## Keywords

`Neural Machine Translation` · `Urdu to English Translation` · `Urdu NLP` · `Low-resource Language Translation` · `BLEU Score` · `HuggingFace Transformers` · `Helsinki-NLP` · `NMT Urdu` · `Translation Model` · `Text Correction Urdu-English` · `OPUS-MT` · `Gradio NLP Demo`

---

## License

This project is open-sourced under the MIT License.

---

## Contributions

Contributions are welcome. Fork the repo, open an issue, or submit a pull request for improvements or new features.

