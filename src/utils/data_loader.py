"""
Tiện ích để tải và xử lý dữ liệu cho RAG pipeline.

Cách dùng:
    from utils.data_loader import load_knowledge_base, split_text, build_vectorstore

    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
    vectorstore = build_vectorstore(chunks, embeddings)
"""
from pathlib import Path


def load_knowledge_base(path: str = None) -> str:
    """
    Đọc file knowledge base và trả về nội dung dạng chuỗi.

    Args:
        path: đường dẫn tới file text.
              Mặc định: data/knowledge_base.txt (thư mục gốc của project)

    Returns:
        Nội dung file dưới dạng str
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "knowledge_base.txt"
    return Path(path).read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Chia văn bản thành các đoạn nhỏ (chunks) để index.

    Dùng RecursiveCharacterTextSplitter — tách ưu tiên theo đoạn văn, câu, rồi ký tự.

    Args:
        text         : văn bản cần chia
        chunk_size   : số ký tự tối đa mỗi chunk (mặc định: 500)
        chunk_overlap: số ký tự chồng lên nhau giữa 2 chunks liên tiếp (mặc định: 50)

    Returns:
        list[str] — danh sách các chuỗi chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def _embed_batches(embeddings, chunks: list, batch_size: int = 32) -> list:
    """
    Embed các chunks theo lô nhỏ, tự retry với backoff khi provider trả 429.

    Cần thiết với các free tier (Gemini: 100 embed-request/phút) — nếu gửi
    một lần toàn bộ chunks thì rất dễ bị rate-limit và hỏng cả pipeline.
    """
    import time

    vectors = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        for attempt in range(1, 7):
            try:
                vectors.extend(embeddings.embed_documents(batch))
                break
            except Exception as e:
                if attempt == 6:
                    raise
                wait = min(60, 15 * attempt)
                print(f"   ⏳ Embedding lỗi ({type(e).__name__}), thử lại sau {wait}s "
                      f"(lần {attempt}/5) ...")
                time.sleep(wait)

        print(f"   • đã embed {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")

    return vectors


def build_vectorstore(chunks: list, embeddings, cache_dir=None):
    """
    Tạo FAISS vectorstore từ danh sách chunks và embeddings.

    Index được cache xuống đĩa (data/faiss_index) để các bước sau tái sử dụng,
    tránh gọi lại API embedding nhiều lần.

    Args:
        chunks    : list[str] — danh sách text chunks đã chia
        embeddings: Embeddings instance (từ get_embeddings())
        cache_dir : thư mục cache index. Mặc định: data/faiss_index

    Returns:
        FAISS vectorstore đã được index và sẵn sàng dùng để retrieve
    """
    from langchain_community.vectorstores import FAISS

    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent.parent / "data" / "faiss_index"
    cache_dir = Path(cache_dir)

    # 1) Dùng lại index đã cache nếu có
    if (cache_dir / "index.faiss").exists():
        try:
            vectorstore = FAISS.load_local(
                str(cache_dir), embeddings, allow_dangerous_deserialization=True
            )
            print(f"♻️  Đã nạp FAISS index từ cache ({cache_dir.name})")
            return vectorstore
        except Exception as e:
            print(f"⚠️  Cache lỗi ({e}) — sẽ index lại từ đầu.")

    # 2) Chưa có cache → embed theo lô rồi build index
    print(f"🔨 Đang tạo FAISS index từ {len(chunks)} chunks ...")
    vectors     = _embed_batches(embeddings, chunks)
    vectorstore = FAISS.from_embeddings(
        list(zip(chunks, vectors)), embeddings
    )

    try:
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(cache_dir))
        print(f"💾 Đã cache index vào {cache_dir}")
    except Exception as e:
        print(f"⚠️  Không cache được index: {e}")

    print("✅ FAISS vectorstore đã sẵn sàng.")
    return vectorstore
