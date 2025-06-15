from sentence_transformers import SentenceTransformer
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig, BaseQAModel, BaseEmbeddingModel, BaseSummarizationModel
import requests
import os
import json
from tqdm import tqdm
import csv

os.environ["OPENAI_API_KEY"] = "not_used"

# TODO: Tokenizer, max_tokens -> see demo.ipynb


class VLLMHTTPQAModel(BaseQAModel):
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url.rstrip("/")

    def answer_question(self, context: str, question: str) -> str:
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        payload = {
            "prompt": prompt,
        }
        resp = requests.post(f"{self.server_url}/v1/completions", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["text"].strip()


class VLLMSummarizationModel(BaseSummarizationModel):
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url.rstrip("/")

    def summarize(self, context: str, max_tokens=150) -> str:
        prompt = (
            f"Please write a concise summary of the following text, "
            f"including all key details:\n\n{context}\n\nSummary:"
        )
        payload = {
            "prompt": prompt,
        }
        resp = requests.post(f"{self.server_url}/v1/completions", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"].strip()


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)


def run(questions_json: str, corpus_json: str, out_csv: str):
    config = RetrievalAugmentationConfig(
        summarization_model=VLLMSummarizationModel(),
        qa_model=VLLMHTTPQAModel(),
        embedding_model=SBertEmbeddingModel(),
    )
    RA = RetrievalAugmentation(config=config)

    corpus = []
    with open(corpus_json, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            corpus.append(json.loads(line))
    documents = str([item["contents"] for item in corpus])
    RA.add_documents(documents)

    questions = []
    with open(questions_json, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            questions.append(json.loads(line))

    results = []
    for ex in tqdm(questions):
        q_id = ex.get("id")
        question = ex["question"]
        gold_list = ex.get("golden_answers", [])
        gold = gold_list[0] if gold_list else ""

        predicted = RA.answer_question(question=question)

        results.append({
            "id":               q_id,
            "question":         question,
            "gold_answer":      gold,
            "predicted_answer": predicted,
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    run(
        questions_json="../../data/hotpotqa/train.jsonl",
        corpus_json="../../data/hotpotqa/corpus_sentence_6.jsonl",
        out_csv="../../results/raptor_hotpotqa.csv",
    )
