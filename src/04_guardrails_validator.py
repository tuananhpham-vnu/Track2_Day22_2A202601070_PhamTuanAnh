"""
Bước 4 — Guardrails AI Validators
====================================
NHIỆM VỤ:
  1. Xây dựng PIIDetector: phát hiện & redact email, số điện thoại, SSN, số thẻ tín dụng
  2. Xây dựng JSONFormatter: tự động sửa JSON lỗi
  3. Bọc mỗi validator trong Guard và test với các mẫu đầu vào
  4. Chạy demo với 6 trường hợp PII và 5 trường hợp JSON

DELIVERABLE: Tất cả test cases pass (PII bị redact, JSON được sửa thành công)

CÁC KHÁI NIỆM CHÍNH:
  - @register_validator     — khai báo custom validator class
  - Validator.validate()    — implement logic kiểm tra + sửa
  - OnFailAction.FIX        — thay thế output thay vì raise error
  - Guard().use(validator)  — gắn validator instance vào guard
  - guard.validate(text)    → ValidationOutcome
      .validation_passed    — bool
      .validated_output     — output đã được xử lý

⚠️  QUAN TRỌNG: on_fail phải truyền vào CONSTRUCTOR của VALIDATOR, KHÔNG phải Guard.use()
    SAI  : Guard().use(PIIDetector, on_fail=OnFailAction.FIX)   ← TypeError
    ĐÚNG : Guard().use(PIIDetector(on_fail=OnFailAction.FIX))   ← correct

Cách chạy:
    python 04_guardrails_validator.py          # chạy cả 2 demo
    python 04_guardrails_validator.py pii      # chỉ demo PII
    python 04_guardrails_validator.py json     # chỉ demo JSON
"""

import re
import sys
import json

from guardrails import Guard
from guardrails.validators import Validator, register_validator, PassResult, FailResult

try:
    from guardrails.hub import OnFailAction
except ImportError:
    from guardrails.validator_base import OnFailAction


# ── 1. PII Detector Validator ──────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Phát hiện và redact Personally Identifiable Information (PII).

    Các pattern được phát hiện:
      EMAIL       : xxx@xxx.xxx
      PHONE       : (123) 456-7890 hoặc 123-456-7890
      SSN         : 123-45-6789
      CREDIT_CARD : 1234 5678 9012 3456 (hoặc dấu gạch nối)
    """

    # Regex patterns cho từng loại PII
    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        # (555) 867-5309  |  +1 555-123-4567  |  555.123.4567
        "PHONE":       r"(?:\+?1[-.\s]?)?\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}"
                       r"|\b(?:\+?1[-.\s]?)?\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        """
        Tìm PII trong value bằng regex; nếu phát hiện thì trả về FailResult kèm
        fix_value là văn bản đã được che — với on_fail=FIX, Guard sẽ thay thế
        output gốc bằng chuỗi an toàn này.
        """
        redacted_text = value
        found_pii     = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, redacted_text)
            if not matches:
                continue

            # Thay thế toàn bộ match của loại PII này bằng placeholder
            redacted_text = re.sub(pattern, f"[{pii_type}_REDACTED]", redacted_text)
            for match in matches:
                found_pii.append((pii_type, match))

        if found_pii:
            types = sorted({p[0] for p in found_pii})
            print(f"  ⚠️  Đã redact {len(found_pii)} PII: {types}")
            return FailResult(
                error_message=f"Phát hiện PII: {', '.join(types)}",
                fix_value=redacted_text,
            )

        return PassResult()


# ── 2. JSON Formatter Validator ────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validate và tự động sửa JSON lỗi.

    Các lỗi có thể sửa tự động:
      - Strip markdown code fences (``` hoặc ```json)
      - Thay single quotes → double quotes
      - Xóa trailing commas trước } hoặc ]
      - Re-serialize với json.dumps để định dạng chuẩn

    Lưu ý: override_value_on_pass = True để Guard áp dụng value_override
    của PassResult (nếu không, output trả về sẽ là chuỗi gốc chưa sửa).
    """

    override_value_on_pass = True   # cho phép Guard áp dụng value_override khi PassResult

    @staticmethod
    def _repair(text: str) -> str:
        """Cố gắng sửa chuỗi JSON lỗi và trả về chuỗi đã sửa (chưa parse)."""
        text = text.strip()

        # 1) Xóa markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$',          '', text)
        text = text.strip()

        # 2) Thay single quotes → double quotes
        text = text.replace("'", '"')

        # 3) Xóa trailing commas trước } hoặc ]
        text = re.sub(r',\s*([}\]])', r'\1', text)

        return text

    def validate(self, value: str, metadata: dict):
        """
        Thử parse value thành JSON; nếu thất bại thì gọi _repair() rồi thử lại.
        Nếu vẫn không parse được → FailResult kèm fix_value là JSON lỗi dự phòng.
        """
        # a) Thử parse trực tiếp — JSON đã hợp lệ, giữ nguyên
        try:
            json.loads(value)
            return PassResult()
        except json.JSONDecodeError:
            pass

        # b) Thử sửa rồi parse lại → trả về FailResult kèm fix_value
        #    (với on_fail=FIX, Guard sẽ thay output gốc bằng JSON đã sửa)
        try:
            repaired_text = self._repair(value)
            parsed        = json.loads(repaired_text)
            print("  🔧 JSON đã được sửa thành công")
            return FailResult(
                error_message="JSON sai định dạng — đã tự động sửa",
                fix_value=json.dumps(parsed, indent=2, ensure_ascii=False),
            )
        except json.JSONDecodeError as e:
            # c) Không sửa được → trả về JSON lỗi chuẩn làm fallback
            fallback = json.dumps(
                {"error": "Không thể phân tích JSON", "raw": value[:200]},
                indent=2, ensure_ascii=False,
            )
            print("  ❌ Không sửa được — dùng JSON lỗi dự phòng")
            return FailResult(
                error_message=f"JSON không hợp lệ sau khi sửa: {e}",
                fix_value=fallback,
            )


# ── 3. Demo: PII Guard ─────────────────────────────────────────────────────
def demo_pii_guard():
    """Chạy 6 test case qua Guard gắn PIIDetector (on_fail=FIX)."""
    print("\n" + "=" * 55)
    print("  Demo: PII Detection & Redaction")
    print("=" * 55)

    # on_fail truyền vào CONSTRUCTOR của validator, không phải Guard.use()
    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",        "Contact John at john.doe@example.com for details."),
        ("Phone",        "Call our support line at (555) 867-5309."),
        ("SSN",          "Patient SSN is 123-45-6789 on file."),
        ("Credit Card",  "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",    "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",        "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        was_fixed = str(result.validated_output) != text

        print(f"\n[{label}] {'🛡️  FIXED (đã che PII)' if was_fixed else '✅ CLEAN (không có PII)'}")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


# ── 4. Demo: JSON Guard ────────────────────────────────────────────────────
def demo_json_guard():
    """Chạy 5 test case qua Guard gắn JSONFormatter (on_fail=FIX)."""
    print("\n" + "=" * 55)
    print("  Demo: JSON Formatting & Repair")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",       '{"name": "Alice", "age": 30}'),
        ("Markdown fences",  '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",    "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",   '{"key": "value",}'),
        ("Truly invalid",    "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result   = guard.validate(text)
        output   = str(result.validated_output)
        is_error = '"error"' in output and "Không thể phân tích JSON" in output

        if is_error:
            status = "🛡️  FALLBACK (JSON lỗi dự phòng)"
        elif output.strip() != text.strip():
            status = "🔧 REPAIRED (đã sửa & chuẩn hoá)"
        else:
            status = "✅ PASS (JSON hợp lệ sẵn)"

        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:60]}")
        print(f"  Output: {output[:200]}")


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    """Chạy demo theo tham số dòng lệnh: pii | json | (mặc định) cả hai."""
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    print("=" * 55)
    print("  Bước 4: Guardrails AI Validators")
    print("=" * 55)

    if which in ("all", "pii"):
        demo_pii_guard()
    if which in ("all", "json"):
        demo_json_guard()

    print("\n✅ Bước 4 hoàn thành!")


if __name__ == "__main__":
    main()
