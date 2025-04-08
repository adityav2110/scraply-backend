from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
import requests
import nltk
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")  # 👈 This fixes the error you're seeing
nltk.download("stopwords")
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import syllapy
from collections import Counter

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "Text Analyzer API is running"

@app.route("/analyze", methods=["POST"])
def analyze_text():
    data = request.get_json()
    url = data.get("url")

    if not url:
        return jsonify({"error": "URL is required"}), 400

    try:
        response = requests.get(url, timeout=10)
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')

        for tag in soup(["script", "style"]):
            tag.extract()

        text = ' '.join(soup.get_text().split())

        tag_counts = Counter([tag.name for tag in soup.find_all()])
        html_tags = [{"tag": tag, "count": count} for tag, count in tag_counts.items()]

        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))

        words_filtered = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha()]

        blob = TextBlob(text)
        positive_score = sum(1 for sentence in blob.sentences if sentence.sentiment.polarity > 0)
        negative_score = sum(1 for sentence in blob.sentences if sentence.sentiment.polarity < 0)
        polarity_score = blob.sentiment.polarity
        subjectivity_score = blob.sentiment.subjectivity

        avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / max(len(sentences), 1)

        complex_words = [word for word in words if syllapy.count(word) > 2]
        percentage_complex_words = len(complex_words) / max(len(words), 1) * 100
        fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
        avg_words_per_sentence = len(words) / max(len(sentences), 1)
        complex_word_count = len(complex_words)
        word_count = len(words_filtered)
        syllables_per_word = sum(syllapy.count(word) for word in words) / max(len(words), 1)
        personal_pronouns = ['i', 'me', 'my', 'we', 'our', 'ours']
        personal_pronoun_count = sum(1 for word in words if word.lower() in personal_pronouns)
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)

        word_freq = Counter(words_filtered)
        least_common = word_freq.most_common()[:-6:-1]
        least_frequent_words = [{"word": word, "count": count} for word, count in least_common]

        result = {
            "metrics": {
                "POSITIVE SCORE": round(positive_score, 2),
                "NEGATIVE SCORE": round(negative_score, 2),
                "POLARITY SCORE": round(polarity_score, 2),
                "SUBJECTIVITY SCORE": round(subjectivity_score, 2),
                "AVG SENTENCE LENGTH": round(avg_sentence_length, 2),
                "PERCENTAGE OF COMPLEX WORDS": round(percentage_complex_words, 2),
                "FOG INDEX": round(fog_index, 2),
                "AVG NUMBER OF WORDS PER SENTENCE": round(avg_words_per_sentence, 2),
                "COMPLEX WORD COUNT": round(complex_word_count, 2),
                "WORD COUNT": round(word_count, 2),
                "SYLLABLE PER WORD": round(syllables_per_word, 2),
                "PERSONAL PRONOUNS": round(personal_pronoun_count, 2),
                "AVG WORD LENGTH": round(avg_word_length, 2)
            },
            "html_tag_frequency": html_tags,
            "least_frequent_words": least_frequent_words
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
