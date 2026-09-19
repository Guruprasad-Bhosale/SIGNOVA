# ISLTranslate Translation Text Analysis

Quantitative analysis of English translation text distributions and linguistic properties.

---

## 1. Summary Statistics

| Metric | Measured Value |
|---|---|
| **Total Samples Analyzed** | 31,222 |
| **Valid Sentence Count** | 31,220 |
| **Unique Sentence Strings** | 28,395 (90.95% uniqueness) |
| **Total Words / Tokens** | 210,223 |
| **Vocabulary Size (Unique Lowercase Words)** | **11,811** |
| **Tokens Per Sentence (Mean $\pm$ Std)** | $6.73 \pm 5.12$ words |
| **Tokens Per Sentence (Median)** | **6 words** |
| **Tokens Per Sentence (Range)** | Min: 0 (empty) to Max: 103 words |
| **95th Percentile Token Length** | 16 words |
| **99th Percentile Token Length** | 23 words |
| **Character Length (Mean / Median / P95)** | 35.15 / 30 / 84 chars |

---

## 2. Vocabulary & Word Frequency Distribution

- **Top 10 Words**: `the` (13,276), `a` (5,969), `of` (5,387), `to` (4,912), `and` (4,872), `in` (4,221), `is` (3,027), `you` (2,683), `it` (1,619), `for` (1,584).
- **Domain Artifacts**: Word `page` appears 1,222 times (due to textbook narration prompts such as "Page 111").
- **Rare Words Distribution**:
  - **Singletons (occur once)**: 4,590 words (38.86% of vocabulary).
  - **Low Frequency ($\le 5$ occurrences)**: 8,497 words (71.94% of vocabulary).
  - **Engineering Implication**: Subword tokenization (e.g. Byte-Pair Encoding / WordPiece with vocab size ~4,000–8,000) is strongly recommended over pure word-level vocabularies to handle rare words gracefully.

---

## 3. Duplicate Sentence Strings

- **Unique Sentences**: 28,395
- **Repeated Sentence Texts**: 1,308 distinct strings appear more than once (totaling 4,133 sample instances).
- **Most Common Repeated Text Examples**: "Page [X]", "Discuss with your partner", "What do you see?", "Let us read".
- **Significance**: In sign language translation datasets, identical English sentences signed by different signers or in different contexts represent natural multi-speaker diversity rather than erroneous duplication.
