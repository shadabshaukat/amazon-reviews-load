from sentence_transformers import SentenceTransformer
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert text to 768-dim vector embedding using nomic-embed-text-v1.5")
    parser.add_argument('--text', '-t', required=True, help="Text string to embed")
    args = parser.parse_args()

    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    vec = model.encode([args.text])[0]

    # Output as comma-separated string for SQL ARRAY[] usage
    print(', '.join(str(float(f)) for f in vec))

if __name__ == "__main__":
    main()
