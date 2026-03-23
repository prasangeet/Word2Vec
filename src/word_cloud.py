import os
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

"""
we read the clean corpus file that was saved by the preprocessing pipeline
and generate a word cloud from the most frequent words
"""

CLEAN_CORPUS = "clean_corpus.txt"
OUTPUT_PATH  = "visualization_wordcloud.png"

"""
we read all the tokens from the clean corpus
each line is one document so we split by whitespace to get the words
"""
all_words = []
with open(CLEAN_CORPUS, "r", encoding="utf-8") as f:
    for line in f:
        all_words.extend(line.strip().split())

print(f"Total words loaded: {len(all_words)}")

"""
we count word frequencies and generate the word cloud
higher frequency words appear larger in the image
"""
word_freq = Counter(all_words)

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    max_words=150,
    colormap="viridis",
    collocations=False
).generate_from_frequencies(word_freq)

plt.figure(figsize=(14, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("IIT Jodhpur Corpus -- Most Frequent Words", fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close()

print(f"Word cloud saved to: {OUTPUT_PATH}")

