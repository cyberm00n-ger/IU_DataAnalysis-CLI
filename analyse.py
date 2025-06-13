from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import torch
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from wordcloud import WordCloud
import ollama

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
# Retro-CLI Prompt for Dataset Size
# ---------------------------------------------------------------------------

def get_dataset_size(total_rows: int) -> int:
    print("\n" + "=" * 50)
    print("   *** IU Topic Modeling System v1.0 ***")
    print("   ╔════════════════════════════════════╗")
    print("   ║        DATASET SELECTION          ║")
    print("   ╚════════════════════════════════════╝")
    print(f"   Available Complaints: {total_rows}")
    print("   Please enter the desired number")
    print("   (or press Enter for default: 50000)")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("   COUNT> ").strip()
            if user_input == "":
                return min(50000, total_rows)
            num_samples = int(user_input)
            if num_samples <= 0:
                print("   ERROR: Number must be positive!")
                continue
            if num_samples > total_rows:
                print(f"   ERROR: Maximum {total_rows} complaints available!")
                continue
            return num_samples
        except ValueError:
            print("   ERROR: Please enter a valid number!")

# ---------------------------------------------------------------------------
# Preprocessing – ONE PIPELINE FOR ALL
# ---------------------------------------------------------------------------

DEFAULT_STOPWORDS: set[str] = {
    "complaint",
    "issue",
    "consumer",
    "from",
    "or",
    "other",
    "my",
    "by",
    "when",
    "you",
    "not",
    "on",
    "and",
    "of",
    "the",
    "'s",
    "cont'd",
    "improper",
    "caused",
}

def preprocess_texts(texts: List[str], nlp, stopwords: set[str] | None = None) -> Tuple[List[str], List[List[str]]]:
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    cleaned_strings: List[str] = []
    token_lists: List[List[str]] = []

    for doc in tqdm(
        nlp.pipe(texts, batch_size=512, n_process=1),
        total=len(texts),
        desc="Preprocessing Texts",
    ):
        tokens = [
            tok.lemma_.lower()
            for tok in doc
            if not (
                tok.is_punct
                or tok.is_digit
                or tok.is_stop
                or tok.lemma_.lower() in stopwords
            )
        ]
        token_lists.append(tokens)
        cleaned_strings.append(" ".join(tokens))

    # Filter empty documents
    valid_pairs = [(s, t) for s, t in zip(cleaned_strings, token_lists) if t]
    if not valid_pairs:
        raise ValueError("No valid documents found after preprocessing.")
    cleaned_strings, token_lists = zip(*valid_pairs)
    return list(cleaned_strings), list(token_lists)

# ---------------------------------------------------------------------------
# Word-Cloud Helper
# ---------------------------------------------------------------------------

def plot_topic_wordcloud(word_weight_pairs: dict[str, float], title: str, out_dir: str = "wordclouds") -> Path:
    os.makedirs(out_dir, exist_ok=True)
    wc = WordCloud(width=800, height=400, random_state=42)
    wc.generate_from_frequencies(word_weight_pairs)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)
    file_path = Path(out_dir) / f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close()
    return file_path

# ---------------------------------------------------------------------------
# LLM Summarization
# ---------------------------------------------------------------------------

def summarize_top_topics(
    topics: List[List[str]],
    weights: np.ndarray | None = None,
    model_name: str = "llama3:latest",
) -> str:
    """Returns a one-sentence summary of the three most important topics."""
    if weights is not None:
        top_idx = weights.argsort()[-3:][::-1]
        top_topics = [topics[i] for i in top_idx]
    else:
        top_topics = topics[:3]

    topics_str = "; ".join([", ".join(t) for t in top_topics])
    prompt = (
        "Summarize in one concise sentence the main themes represented by these topic words: "
        f"{topics_str}. Focus on the most prominent issues in consumer financial complaints."
    )

    try:
        resp = ollama.generate(model=model_name, prompt=prompt, options={"temperature": 0.3})
        return resp["response"].strip()
    except Exception as exc:
        return f"LLM error: {exc}"

# ---------------------------------------------------------------------------
# Topic Modeling Helpers
# ---------------------------------------------------------------------------

def fit_and_score(
    model,
    vec,
    matrix,
    texts: List[List[str]],
    dictionary: Dictionary,
    use_abs: bool = False,
) -> Tuple[float, List[List[str]]]:
    """Trains *model* and computes C_v coherence."""
    model.fit(matrix)
    features = vec.get_feature_names_out()
    topics = [
        [features[i] for i in (np.abs(comp) if use_abs else comp).argsort()[-5:][::-1]]
        for comp in tqdm(model.components_, desc=f"Extracting Topics for {model.__class__.__name__}")
    ]
    cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence="c_v", processes=1)
    return cm.get_coherence(), topics

# ---------------------------------------------------------------------------
# Word2Vec + Clustering
# ---------------------------------------------------------------------------

def train_word2vec_and_cluster(
    token_lists: List[List[str]],
    n_clusters: int = 10,
) -> Tuple[KMeans, float, List[List[str]]]:
    # Word2Vec training with progress bar
    w2v = Word2Vec(sentences=tqdm(token_lists, desc="Training Word2Vec"), vector_size=100, window=5, min_count=10, workers=4, seed=42)
    doc_vectors = [
        np.mean([w2v.wv[w] for w in doc if w in w2v.wv], axis=0)
        if any(w in w2v.wv for w in doc)
        else np.zeros(100)
        for doc in tqdm(token_lists, desc="Computing Document Vectors")
    ]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(doc_vectors)
    labels = kmeans.labels_

    topics: List[List[str]] = []
    dictionary = Dictionary(token_lists)

    for cl in tqdm(range(n_clusters), desc="Extracting Word2Vec Topics"):
        words = [w for idx, doc in enumerate(token_lists) if labels[idx] == cl for w in doc]
        word_counts = pd.Series(words).value_counts()
        valid_words = [w for w in word_counts.head(5).index if w in dictionary.token2id]
        topics.append(valid_words if valid_words else ["unknown"])

    unknown_count = sum(1 for t in topics if t == ["unknown"])
    if unknown_count > 3:
        print(f"Warning: {unknown_count} of {n_clusters} Word2Vec topics are invalid. Consider increasing dataset size or cluster count.")

    cm = CoherenceModel(topics=topics, texts=token_lists, dictionary=dictionary, coherence="c_v", processes=1)
    return kmeans, cm.get_coherence(), topics

# ---------------------------------------------------------------------------
# BERT + Clustering
# ---------------------------------------------------------------------------

def get_bert_embeddings_and_cluster(
    cleaned_texts: List[str],
    n_clusters: int = 10,
) -> Tuple[KMeans, float, List[List[str]]]:
    """Generates [CLS] embeddings and clusters them."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()

    embeddings: List[np.ndarray] = []
    for i in tqdm(range(0, len(cleaned_texts), 32), desc="Computing BERT Embeddings"):
        batch = cleaned_texts[i : i + 32]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outs = model(**inputs)
        embeddings.append(outs.last_hidden_state[:, 0, :].cpu().numpy())

    doc_vecs = np.vstack(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(doc_vecs)  
    labels = kmeans.labels_

    token_lists = [t.split() for t in cleaned_texts]
    dictionary = Dictionary(token_lists)
    topics: List[List[str]] = []

    for cl in tqdm(range(n_clusters), desc="Extracting BERT Topics"):
        words = [w for idx, doc in enumerate(token_lists) if labels[idx] == cl for w in doc]
        word_counts = pd.Series(words).value_counts()
        valid_words = [w for w in word_counts.head(5).index if w in dictionary.token2id]
        topics.append(valid_words if valid_words else ["unknown"])

    unknown_count = sum(1 for t in topics if t == ["unknown"])
    if unknown_count > 3:
        print(f"Warning: {unknown_count} of {n_clusters} BERT topics are invalid. Consider increasing dataset size or cluster count.")

    cm = CoherenceModel(topics=topics, texts=token_lists, dictionary=dictionary, coherence="c_v", processes=1)
    return kmeans, cm.get_coherence(), topics

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main(path: str = "data.csv", seed: int = 42) -> None:
    set_seed(seed)
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])

    # --- Load data & user prompt for count
    df = pd.read_csv(path, usecols=["issue"])
    total_rows = len(df)
    num_samples = get_dataset_size(total_rows)
    complaints = random.sample(df["issue"].dropna().astype(str).tolist(), k=num_samples)
    complaints = list(tqdm([complaints], desc="Sampling Complaints"))[0]  

    # --- Unified preprocessing
    cleaned, token_lists = preprocess_texts(complaints, nlp)
    dictionary = Dictionary(token_lists)

    # --- Bag-of-Words / TF-IDF vectorization
    bow_vec = CountVectorizer(max_features=5_000, min_df=15, max_df=0.7)
    bow = bow_vec.fit_transform(cleaned)  

    tfidf_vec = TfidfVectorizer(max_features=5_000, min_df=15, max_df=0.7)
    tfidf = tfidf_vec.fit_transform(cleaned)  

    # --- Classical models
    classical = {
        "LDA": {
            "model": LatentDirichletAllocation(n_components=10, random_state=42, n_jobs=-1, learning_method="online"),
            "vec": bow_vec,
            "matrix": bow,
            "abs": False,
        },
        "LSA": {
            "model": TruncatedSVD(n_components=10, random_state=42),
            "vec": tfidf_vec,
            "matrix": tfidf,
            "abs": True,
        },
        "NMF": {
            "model": NMF(n_components=10, random_state=42, init="nndsvd", max_iter=400),
            "vec": tfidf_vec,
            "matrix": tfidf,
            "abs": False,
        },
    }

    results: dict[str, dict] = {}

    for name, cfg in tqdm(classical.items(), desc="Training Classical Models"):
        score, topics = fit_and_score(cfg["model"], cfg["vec"], cfg["matrix"], token_lists, dictionary, cfg["abs"])
        results[name] = {
            "model": cfg["model"],
            "score": score,
            "topics": topics,
            "vec": cfg["vec"],
            "matrix": cfg["matrix"],
            "abs": cfg["abs"],
        }

    # --- Word2Vec + BERT (semantic)
    w2v_model, w2v_score, w2v_topics = train_word2vec_and_cluster(token_lists)
    results["Word2Vec"] = {"model": w2v_model, "score": w2v_score, "topics": w2v_topics, "matrix": None}

    bert_model, bert_score, bert_topics = get_bert_embeddings_and_cluster(cleaned)
    results["BERT"] = {"model": bert_model, "score": bert_score, "topics": bert_topics, "matrix": None}

    # --- Word clouds for the two best models
    print("\n===== Generating Word Clouds =====")
    for name, res in tqdm(sorted(results.items(), key=lambda kv: kv[1]["score"], reverse=True)[:2], desc="Generating Word Clouds"):
        for idx, twords in enumerate(tqdm(res["topics"], desc=f"Creating Word Clouds for {name}")):
            freqs: dict[str, float]
            if name in {"Word2Vec", "BERT"}:
                freqs = {w: 1.0 / (i + 1) for i, w in enumerate(twords)}
            else:
                comps = res["model"].components_[idx]
                idxs = np.abs(comps).argsort()[-5:][::-1]
                freqs = {res["vec"].get_feature_names_out()[j]: (abs(comps[j]) if res["abs"] else comps[j]) for j in idxs}
            plot_topic_wordcloud(freqs, f"{name} Topic {idx}")

    # --- Output
    print("\n===== Topic Modeling Results =====")
    for name, res in results.items():
        print(f"\n{name}: Coherence {res['score']:.4f}, 10 Topics")
        for i, t in enumerate(res["topics"]):
            print(f"  Topic {i:<2} → {', '.join(t)}")

    # --- LLM summaries (top-3 models by score)
    print("\n===== LLM Summary of Key Themes =====")
    top3 = sorted(results.items(), key=lambda kv: kv[1]["score"], reverse=True)[:3]
    for name, res in top3:  # tqdm removed
        weights = None
        if res["matrix"] is not None:
            weights = res["model"].transform(res["matrix"]).sum(axis=0)
        summary = summarize_top_topics(res["topics"], weights)
        print(f"{name}: {summary}")

    # --- Summary
    print("\n===== Summary =====")
    for name, res in results.items():
        print(f"{name:8}: Coherence {res['score']:.4f}, 10 Topics")

# ---------------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Topic Modeling for CFPB Complaints")
    parser.add_argument("--csv", default="data.csv", help="Path to CSV file with 'issue' column")
    parser.add_argument("--seed", type=int, default=42, help="RNG Seed")
    args = parser.parse_args()

    main(path=args.csv, seed=args.seed)