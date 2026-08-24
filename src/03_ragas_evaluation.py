"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Chỉ dùng context bên dưới để trả lời.\n"
    "Hãy trả lời thật ngắn gọn và trực tiếp trong 2-4 câu, dùng ngôn ngữ đời thường, "
    "không dùng bullet point và không nhắc lại câu hỏi.\n"
    "IMPORTANT: Always write the answer in the SAME LANGUAGE as the question "
    "(English question -> English answer).\n"
    "Nếu context không có thông tin, hãy nói thẳng là bạn không biết.\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích tài liệu kỹ thuật. Hãy đọc kỹ context bên dưới, "
    "xác định các dữ kiện liên quan, rồi viết câu trả lời có cấu trúc trong 3-5 câu theo trình tự:\n"
    "  1) Định nghĩa/kết luận chính,\n"
    "  2) Các chi tiết hỗ trợ được trích từ context,\n"
    "  3) Mức độ chắc chắn dựa trên context.\n"
    "Dùng thuật ngữ chuyên môn chính xác và TUYỆT ĐỐI không suy đoán ngoài context. "
    "IMPORTANT: Always write the answer in the SAME LANGUAGE as the question "
    "(English question -> English answer).\n"
    "Nếu context không đủ dữ kiện, hãy nêu rõ phần nào còn thiếu.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}

# RunConfig — hạn chế song song + retry để tránh rate-limit của provider free tier
RUN_CONFIG = RunConfig(max_workers=4, timeout=300, max_retries=10, max_wait=60)


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs     = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]     # phải là list[str] !
    ctx_str  = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(get_vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.

    Kết quả được cache vào data/rag_outputs_<version>.json — nếu chạy lại
    (ví dụ sau khi bị ngắt giữa chừng) thì không cần gọi lại LLM.

    get_vectorstore là callable — chỉ được gọi khi thật sự phải chạy RAG,
    nhờ vậy lần chạy dùng cache không tốn thêm request build FAISS index.

    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    cache_path = Path(__file__).parent.parent / "data" / f"rag_outputs_{prompt_version}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if len(cached) == len(QA_PAIRS):
                print(f"♻️  Dùng lại {len(cached)} kết quả RAG đã cache cho {prompt_version}")
                return cached
        except Exception as e:
            print(f"⚠️  Cache {cache_path.name} lỗi ({e}) — sẽ chạy lại.")

    retriever = get_vectorstore().as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy {len(QA_PAIRS)} câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        try:
            out = run_rag(retriever, llm, prompt, qa["question"])
        except Exception as e:
            print(f"  ⚠️  Lỗi ở câu {i}: {e}")
            out = {"answer": "Tôi không tìm thấy thông tin này trong context.", "contexts": []}

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],     # list[str] !
        })
        preview = qa["question"][:60]
        print(f"  [{i:02d}/{len(QA_PAIRS)}] {preview}")

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"⚠️  Không lưu được cache: {e}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


def _metric_values(result, key: str) -> list:
    """Lấy list điểm của một metric từ EvaluationResult (hỗ trợ nhiều phiên bản RAGAS)."""
    try:
        return list(result[key])
    except Exception:
        return result.to_pandas()[key].tolist()


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = LangchainLLMWrapper(get_llm(temperature=0))
    emb_eval = LangchainEmbeddingsWrapper(get_embeddings())

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=RUN_CONFIG,
    )

    # Tính mean score cho mỗi metric (bỏ qua các sample bị NaN/None)
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw   = _metric_values(result, key)
        valid = [float(v) for v in raw
                 if v is not None and not (isinstance(v, float) and np.isnan(v))]
        scores[key] = float(np.mean(valid)) if valid else 0.0

    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    # Vectorstore chỉ được build khi cache RAG outputs chưa có (lazy)
    _vs_cache = {}

    def get_vectorstore():
        if "vs" not in _vs_cache:
            _vs_cache["vs"] = setup_vectorstore()
        return _vs_cache["vs"]

    # Thu thập kết quả RAG cho cả V1 và V2
    v1_results = collect_rag_outputs(get_vectorstore, "v1")
    v2_results = collect_rag_outputs(get_vectorstore, "v2")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # Lưu báo cáo vào data/ragas_report.json
    report = {
        "num_samples":      len(QA_PAIRS),
        "provider":         config.PROVIDER,
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met":       best_faith >= 0.8,
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False)

    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(payload, encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")

    # Sao chép sang evidence/ để nộp bài
    evidence_path = Path(__file__).parent.parent / "evidence" / "03_ragas_report.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(payload, encoding="utf-8")
    print(f"💾 Đã sao chép vào {evidence_path}")


if __name__ == "__main__":
    main()
