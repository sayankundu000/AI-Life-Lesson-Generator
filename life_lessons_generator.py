import os
import argparse
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import glob

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch is not installed. Using fallback embeddings.")
    TORCH_AVAILABLE = False

try:
    if TORCH_AVAILABLE:
        from sentence_transformers import SentenceTransformer
    else:
        raise ImportError
except ImportError:
    print("⚠️ sentence-transformers not available. Using lightweight fallback embedder.")
    class SentenceTransformer:
        def __init__(self, model_name='fallback'):
            print(f"Using fallback embedding model instead of '{model_name}'")

        def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
            if isinstance(texts, str):
                texts = [texts]
            embeddings = []
            for text in texts:
                vec = np.zeros(300)
                for word in text.split():
                    for i, ch in enumerate(word.encode('utf-8')):
                        vec[i % 300] += ch / 255.0
                embeddings.append(vec / max(1, len(text.split())))
            return np.array(embeddings)

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ Transformers not available. Using dummy text generation.")
    TRANSFORMERS_AVAILABLE = False

    class DummyPipeline:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, prompt, max_length=1300, num_return_sequences=1, truncation=True):
            return [{'generated_text': f"{prompt} [Dummy life lesson generated because Transformers is unavailable]"}]

    pipeline = DummyPipeline

DATA_PATH = './data/*.txt'
EMBED_MODEL = 'all-MiniLM-L6-v2'
GEN_MODEL = 'gpt2'
EMBED_FILE = 'embeddings.npy'
CHUNKS_FILE = 'chunks.npy'
TOPICS_FILE = 'topics.npy'
TOKEN_LIMIT = 1700

def generate_api_key():
    return 'AI-LIFELESSON-' + ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=24))
API_KEY = generate_api_key()

def load_corpus():
    files = glob.glob(DATA_PATH)
    corpus = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read().strip()
                if text:
                    corpus.append(text)
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")
    return corpus

def chunk_text(text, max_len=300):
    words = text.split()
    if not words:
        return []
    return [' '.join(words[i:i+max_len]) for i in range(0, len(words), max_len)]

def build_index():
    print("Building index... 🧠")
    corpus = load_corpus()
    if not corpus:
        raise RuntimeError("No text files found in ./data. Add .txt files and re-run with --build")

    all_chunks = []
    for doc in corpus:
        all_chunks.extend(chunk_text(doc))

    if not all_chunks:
        raise RuntimeError("No text content found in the provided files after chunking.")

    print("Encoding passages...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)

    print("Fitting TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(all_chunks)

    n_samples = len(all_chunks)
    n_components = min(50, n_samples - 1) if n_samples > 1 else 1
    svd = TruncatedSVD(n_components=n_components)
    reduced = svd.fit_transform(tfidf_matrix)

    n_clusters = min(10, n_samples)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(reduced)

    np.save(EMBED_FILE, embeddings)
    np.save(CHUNKS_FILE, np.array(all_chunks, dtype=object))
    np.save(TOPICS_FILE, kmeans.labels_)

    print("Index built and saved! ✅")

def _ensure_index_exists():
    if not (os.path.exists(EMBED_FILE) and os.path.exists(CHUNKS_FILE)):
        print("⚠️ Index files not found. Automatically building index...")
        build_index()

def retrieve_relevant_chunks(query, top_k=3):
    _ensure_index_exists()
    model = SentenceTransformer(EMBED_MODEL)
    q_emb = model.encode([query], convert_to_numpy=True)
    chunks = np.load(CHUNKS_FILE, allow_pickle=True)
    emb = np.load(EMBED_FILE)
    sims = cosine_similarity(emb, q_emb).squeeze()
    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(-sims)[:top_k]
    return [chunks[int(i)] for i in top_indices]

def generate_lesson(query):
    print(f"\n🔍 Query: {query}")
    relevant_chunks = retrieve_relevant_chunks(query)
    context = '\n'.join(relevant_chunks)

    generator = pipeline('text-generation', model=GEN_MODEL)
    prompt = f"From sacred texts and wise sayings, derive a short life lesson about '{query}'. Context: {context}\nLife Lesson:"
    output = generator(prompt, max_length=1300, num_return_sequences=1, truncation=True)
    generated = output[0].get('generated_text', '')

    if 'Life Lesson:' in generated:
        generated = generated.split('Life Lesson:')[-1].strip()
    print(f"💡 Life Lesson: {generated}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true', help='Rebuild the data index')
    args = parser.parse_args()

    predefined_queries = ['forgiveness', 'failure', 'purpose']

    if args.build:
        build_index()
    else:
        print("Welcome to the Life Lessons Generator 🌍")
        print(f"API KEY: {API_KEY}")
        print("Generating lessons for predefined queries...\n")
        for query in predefined_queries:
            generate_lesson(query)