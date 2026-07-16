# 인덱싱 진행 상황을 파일(checkpoint.json)로 기록해, 중간에 끊겨도 이어서 처리할 수 있게 하는 모듈
# PDF별로 "몇 페이지까지 처리했는지"를 기록하고, main.py가 재실행 시 이를 읽어 재개 지점을 찾는다.

import json
from pathlib import Path

from config import OUTPUT_DIR

CHECKPOINT_PATH = Path(OUTPUT_DIR) / "checkpoint.json"


def load_checkpoint() -> dict:
    """저장된 checkpoint가 있으면 불러오고, 없으면 빈 상태를 반환한다."""
    if not CHECKPOINT_PATH.exists():
        return {"pdfs": {}}
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(state: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def is_pdf_done(state: dict, pdf_name: str) -> bool:
    """해당 PDF의 모든 페이지 처리가 끝나 인덱스에 반영되었는지 확인한다."""
    entry = state["pdfs"].get(pdf_name)
    return bool(entry and entry.get("status") == "done")


def get_resume_page(state: dict, pdf_name: str) -> int:
    """이 PDF에서 다음에 처리해야 할 0-indexed 페이지 번호를 반환한다 (처음이면 0)."""
    entry = state["pdfs"].get(pdf_name)
    if entry is None:
        return 0
    return entry.get("completed_pages", 0)


def mark_page_done(state: dict, pdf_name: str, page_index: int, total_pages: int) -> None:
    """page_index(0-indexed) 페이지 처리를 완료로 기록하고 즉시 디스크에 저장한다."""
    entry = state["pdfs"].setdefault(pdf_name, {})
    entry["total_pages"] = total_pages
    entry["completed_pages"] = page_index + 1
    entry["status"] = "done" if entry["completed_pages"] >= total_pages else "in_progress"
    save_checkpoint(state)
