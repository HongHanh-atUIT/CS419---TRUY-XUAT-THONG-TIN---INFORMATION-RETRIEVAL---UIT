# ============================================================
# Finetune SBERT với CoSENTLoss trên NFCorpus
# pip install sentence-transformers datasets
# ============================================================

import json
import csv
import os
import random
from collections import defaultdict, Counter

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import CoSENTLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# ==================== CẤU HÌNH ====================
CORPUS_PATH  = "Data/clean_corpus.jsonl"
QUERIES_PATH = "Data/clean_queries.jsonl"
TRAIN_QRELS  = "Data/qrels/train.tsv"
DEV_QRELS    = "Data/qrels/dev.tsv"

BASE_MODEL   = "pritamdeka/S-PubMedBert-MS-MARCO"
OUTPUT_PATH  = "Model/"

EPOCHS       = 3
BATCH_SIZE   = 88
WARMUP_RATIO = 0.1
LR           = 1e-5

# score=2 (directly linked)   → 1.0
# score=1 (indirectly linked) → 0.6
# score=0 (marginal)          → 0.0
SCORE_MAP = {2: 1.0, 1: 0.6, 0: 0.0}


# ==================== HÀM LOAD DỮ LIỆU ====================

def load_jsonl(path: str) -> dict:
    """Load corpus hoặc queries từ file .jsonl → {_id: text}"""
    data = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                data[obj["_id"]] = obj["text"]
    return data


def load_qrels(path: str) -> dict:
    """Load qrels từ file .tsv → {query_id: {doc_id: score}}"""
    qrels = defaultdict(dict)
    if not os.path.exists(path):
        print(f"[WARN] Không tìm thấy: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    return dict(qrels)


# ==================== HÀM XÂY DỰNG TRAINING PAIRS ====================

def build_cosent_examples(qrels_dict, queries, corpus, score_map, split_name=""):
    """
    Tạo InputExample(texts=[query, doc], label=float) cho CoSENTLoss.
    label ánh xạ từ relevance score theo score_map.
    """
    examples = []
    skipped  = 0

    for q_id, doc_scores in qrels_dict.items():
        if q_id not in queries:
            skipped += 1
            continue
        q_text = queries[q_id]

        for doc_id, score in doc_scores.items():
            if doc_id not in corpus:
                skipped += 1
                continue
            label = score_map.get(score, None)
            if label is None:
                continue
            examples.append(InputExample(
                texts=[q_text, corpus[doc_id]],
                label=label
            ))

    print(f"[{split_name}] Examples: {len(examples):,}  |  Skipped: {skipped}")
    return examples


# ==================== HÀM BUILD EVALUATOR ====================

def build_evaluator(dev_examples, name="dev-cosent"):
    """
    Tạo EmbeddingSimilarityEvaluator từ dev examples.
    Theo dõi Spearman correlation giữa cosine similarity và label sau mỗi epoch.
    Dùng để save_best_model theo epoch tốt nhất.
    """
    if not dev_examples:
        return None

    sentences1 = [e.texts[0] for e in dev_examples]
    sentences2 = [e.texts[1] for e in dev_examples]
    labels     = [e.label    for e in dev_examples]

    return EmbeddingSimilarityEvaluator(
        sentences1, sentences2, labels,
        name=name,
        show_progress_bar=False,
        write_csv=True,
    )


# ==================== HÀM FINETUNE ====================

def finetune(model, all_examples, evaluator, output_path,
             epochs, batch_size, warmup_ratio, lr):
    """Finetune SBERT với CoSENTLoss và lưu checkpoint tốt nhất."""
    train_loader = DataLoader(all_examples, shuffle=True, batch_size=batch_size)
    loss_fn      = CoSENTLoss(model)
    warmup_steps = int(len(train_loader) * epochs * warmup_ratio)

    print("\n" + "=" * 60)
    print("FINETUNE SBERT VỚI CoSENTLoss")
    print("=" * 60)
    print(f"  Total examples: {len(all_examples):,}")
    print(f"  Epochs        : {epochs}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Steps/epoch   : {len(train_loader)}")
    print(f"  Warmup steps  : {warmup_steps}")
    print(f"  Learning rate : {lr}")
    print(f"  Output        : {output_path}")
    print("=" * 60)

    os.makedirs(output_path, exist_ok=True)

    model.fit(
        train_objectives=[(train_loader, loss_fn)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=output_path,
        save_best_model=True,
        checkpoint_path=output_path,
        checkpoint_save_steps=len(train_loader),  # lưu mỗi epoch
        checkpoint_save_total_limit=3,            # giữ 3 checkpoint gần nhất
        show_progress_bar=True,
    )

    print(f"\nFinetune hoàn tất! Best model lưu tại: {output_path}")


# ==================== HÀM KIỂM TRA MODEL ====================

def verify_model(output_path):
    """Load lại model đã lưu, test encode và liệt kê files."""
    print("\nLoad lại model từ checkpoint để kiểm tra...")
    model_loaded = SentenceTransformer(output_path)

    test_query = "What is the effect of diet on inflammation?"
    test_doc   = "Dietary interventions can reduce systemic inflammation markers."

    q_vec = model_loaded.encode(test_query, normalize_embeddings=True)
    d_vec = model_loaded.encode(test_doc,   normalize_embeddings=True)
    sim   = float(q_vec @ d_vec)

    print(f"  Query            : {test_query}")
    print(f"  Doc              : {test_doc}")
    print(f"  Cosine similarity: {sim:.4f}")
    print(f"  Model dim        : {model_loaded.get_sentence_embedding_dimension()}")

    print(f"\nFiles trong {output_path}:")
    for fname in sorted(os.listdir(output_path)):
        size = os.path.getsize(os.path.join(output_path, fname))
        print(f"  {fname:<40} {size/1024/1024:.1f} MB" if size > 1024 else f"  {fname}")


# ==================== MAIN ====================

def main():
    print(f"torch : {torch.__version__}")
    print(f"GPU   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. Load dữ liệu
    print("\nLoad corpus & queries...")
    corpus  = load_jsonl(CORPUS_PATH)
    queries = load_jsonl(QUERIES_PATH)
    print(f"  Corpus : {len(corpus):,} documents")
    print(f"  Queries: {len(queries):,} queries")

    print("\nLoad qrels...")
    train_qrels = load_qrels(TRAIN_QRELS)
    dev_qrels   = load_qrels(DEV_QRELS)
    print(f"  Train qrels: {len(train_qrels):,} queries")
    print(f"  Dev qrels  : {len(dev_qrels):,} queries")

    # 2. Xây dựng training pairs
    train_examples = build_cosent_examples(train_qrels, queries, corpus, SCORE_MAP, "train")
    dev_examples   = build_cosent_examples(dev_qrels,   queries, corpus, SCORE_MAP, "dev  ")

    all_examples = train_examples + dev_examples
    random.shuffle(all_examples)
    print(f"\nTổng examples để finetune: {len(all_examples):,}")

    label_counts = Counter(round(e.label, 1) for e in all_examples)
    print("Phân phối label:", dict(sorted(label_counts.items())))

    # 3. Build evaluator
    evaluator = build_evaluator(dev_examples)
    print(f"Evaluator sẵn sàng với {len(dev_examples):,} dev pairs")

    # 4. Load base model
    print(f"\nLoad model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)
    print(f"  Embedding dim  : {model.get_sentence_embedding_dimension()}")
    print(f"  Max seq length : {model.max_seq_length}")

    # 5. Finetune
    finetune(
        model=model,
        all_examples=all_examples,
        evaluator=evaluator,
        output_path=OUTPUT_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        warmup_ratio=WARMUP_RATIO,
        lr=LR,
    )

    # 6. Kiểm tra model đã lưu
    verify_model(OUTPUT_PATH)


if __name__ == "__main__":
    main()