import argparse
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig, BaseQAModel, BaseEmbeddingModel, BaseSummarizationModel
import requests
import os
import json
from tqdm import tqdm
import csv
import logging
os.environ["OPENAI_API_KEY"] = "not_used"

# TODO: Tokenizer, max_tokens -> see demo.ipynb
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class VLLMHTTPQAModel(BaseQAModel):
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url.rstrip("/")

    def answer_question(self, context: str, question: str) -> str:
        system_tmpl = (
            "Answer the question based on the given document. "
            "Return exactly the minimal answer—a named entity or phrase—without any extra words, quotes, or punctuation.\n"
            f"The following are given documents:\n\n{context}"
        )
        user_tmpl = f"Question: {question}"

        payload = {
            "messages": [
                {"role": "system", "content": system_tmpl},
                {"role": "user",   "content": user_tmpl}
            ],
            "temperature": 0.7,
            "top_p":       0.8,
            # vLLM‐only params:
            "extra_body": {
                "top_k": 20,
                "min_p": 0
            },
        }

        resp = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()

        # extract the assistant message
        text = data["choices"][0]["message"]["content"]
        return text.strip()



class VLLMSummarizationModel(BaseSummarizationModel):
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url.rstrip("/")

    def summarize(self, context: str, max_tokens: int = 200) -> str:
        system_tmpl = (
            "You are a helpful assistant."
        )
        user_tmpl = (
            "Write a summary of the following, including as many key details as possible: " + context
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_tmpl},
                {"role": "user",   "content": user_tmpl},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p":       0.8,
            # vLLM‐only params:
            "extra_body": {
                "top_k": 20,
                "min_p": 0
            },
        }

        resp = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()

        text = data["choices"][0]["message"]["content"]
        return text.strip()


class Qwen3EmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="/ukp-storage-1/rolka1/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/80946ea0efeac60523ec1a2cc5a65428a650007e"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)

def build_tree(corpus_jsonl: str, tree_path: Path, config: RetrievalAugmentationConfig):
    logger.info("Building Raptor index tree from corpus: %s", corpus_jsonl)
    
    corpus = []
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            corpus.append(json.loads(line))
    documents = str([item["contents"] for item in corpus])
    
    logger.info("Adding documents to Raptor index")
    
    RA = RetrievalAugmentation(config=config)
    RA.add_documents(documents, corpus_jsonl)

    tree_path.parent.mkdir(parents=True, exist_ok=True)
    RA.save(str(tree_path))
    logger.info("Saved Raptor index tree to %s", tree_path)


def run(questions_jsonl: str, corpus_jsonl: str, out_csv: str, dataset_name: str):
    logger.info("Initializing retrieval‐augmentation pipeline")
    config = RetrievalAugmentationConfig(
        summarization_model=VLLMSummarizationModel(),
        qa_model=VLLMHTTPQAModel(),
        embedding_model=Qwen3EmbeddingModel(),
    )
    
    # check if tree exists
    tree_path = Path(f"/ukp-storage-1/rolka1/thesis/data/{dataset_name}/indexes/raptor_tree.pickle")
    if not tree_path.exists():
        build_tree(corpus_jsonl, tree_path, config)
    
    RA = RetrievalAugmentation(config=config, tree=str(tree_path))

    # Load questions
    questions = []
    with open(questions_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    results = []
    logger.info("Beginning question answering loop")
    for ex in tqdm(questions, desc="QA"):
        q_id = ex.get("id")
        question = ex.get("question", "")
        gold_list = ex.get("golden_answers", [])

        predicted, context = RA.answer_question(question=question, top_k=5)

        results.append({
            "id":      q_id,
            "question": question,
            "gold_answers": gold_list,
            "output": {
                "retrieval_result": context,
                "pred": predicted
            }
        })

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    out_dir = f"/ukp-storage-1/rolka1/thesis/output/{dataset_name}_{timestamp}_raptor"
    os.makedirs(out_dir, exist_ok=True)

    intermediate_path = os.path.join(out_dir, "intermediate_data.json")
    with open(intermediate_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=True)

    print(f"✅ Wrote intermediate data to {intermediate_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="hotpotqa")
    args = parser.parse_args()
    run(
        questions_jsonl=f"/ukp-storage-1/rolka1/thesis/data/{args.dataset_name}/dev.jsonl",
        corpus_jsonl=f"/ukp-storage-1/rolka1/thesis/data/{args.dataset_name}/corpus_sentence_256.jsonl",
        out_csv=f"/ukp-storage-1/rolka1/thesis/output/raptor_{args.dataset_name}.csv",
        dataset_name=args.dataset_name
    )
