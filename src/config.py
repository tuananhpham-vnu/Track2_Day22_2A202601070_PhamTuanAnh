"""
Tải cấu hình từ file .env và thiết lập biến môi trường LangSmith.

⚠️  Import module này TRƯỚC KHI import bất kỳ thư viện LangChain nào.
    config.py tự động set LANGCHAIN_* vào os.environ khi được import.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Tải .env từ thư mục gốc của project (Lab/)
_root = Path(__file__).parent.parent
load_dotenv(_root / ".env")

# ── LangSmith — PHẢI set trước khi import LangChain ──────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "day22-lab")
os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ── Provider mặc định ─────────────────────────────────────────────────────
# Đổi giá trị PROVIDER trong .env:
#   openai | gemini | anthropic | deepseek | ollama | openrouter
PROVIDER = os.getenv("PROVIDER", "openai").lower()

# Provider dùng riêng cho Embeddings. Để trống = dùng chung PROVIDER.
# Cần thiết khi PROVIDER không có Embeddings API (anthropic, deepseek):
#   PROVIDER=deepseek + EMBEDDING_PROVIDER=gemini
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "").lower()

# ── OpenAI ────────────────────────────────────────────────────────────────
OPENAI_API_KEY         = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL        = os.getenv("OPENAI_BASE_URL", "")   # để trống nếu dùng OpenAI chính thức
OPENAI_MODEL           = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ── Google Gemini ─────────────────────────────────────────────────────────
GOOGLE_API_KEY          = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL            = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_EMBEDDING_MODEL  = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

# ── Anthropic ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# ── DeepSeek (API tương thích OpenAI, KHÔNG có Embeddings API) ───────────
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# ── Ollama (local, không cần API key) ────────────────────────────────────
OLLAMA_BASE_URL         = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL            = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_EMBEDDING_MODEL  = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ── OpenRouter ────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── LangSmith ─────────────────────────────────────────────────────────────
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "day22-lab")


def validate() -> bool:
    """
    Kiểm tra các biến môi trường bắt buộc đã được cấu hình.
    Trả về True nếu hợp lệ, False nếu thiếu.
    """
    missing = []

    if not LANGSMITH_API_KEY:
        missing.append("LANGCHAIN_API_KEY (LangSmith)")

    if PROVIDER == "openai" and not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    elif PROVIDER == "gemini" and not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    elif PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    elif PROVIDER == "deepseek" and not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    elif PROVIDER == "openrouter" and not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    # Ollama: không cần API key

    # DeepSeek/Anthropic không có Embeddings API → bắt buộc chỉ định EMBEDDING_PROVIDER
    emb_provider = EMBEDDING_PROVIDER or PROVIDER
    if PROVIDER in ("deepseek", "anthropic") and emb_provider == PROVIDER:
        missing.append(
            f"EMBEDDING_PROVIDER ({PROVIDER} không có Embeddings API — "
            "đặt EMBEDDING_PROVIDER=gemini hoặc openai)"
        )
    elif emb_provider == "gemini" and not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY (dùng cho EMBEDDING_PROVIDER=gemini)")
    elif emb_provider in ("openai", "openrouter") and not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY (dùng cho EMBEDDING_PROVIDER=openai)")

    if missing:
        print("⚠️  Thiếu biến môi trường:")
        for m in missing:
            print(f"   - {m}")
        print("   Hãy kiểm tra file .env của bạn (xem .env.example để biết thêm).")
        return False

    print(f"✅ Config OK  |  LLM: {PROVIDER.upper()}  |  "
          f"Embeddings: {emb_provider.upper()}  |  Project: {LANGSMITH_PROJECT}")
    return True


if __name__ == "__main__":
    validate()
