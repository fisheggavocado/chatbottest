# PDF 페이지를 이미지로 렌더링하고 gpt-5-mini(비전 모델)로 텍스트를 추출하는 저수준 유틸리티
# LlamaIndex 연동은 llama_pdf_reader.py에서 이 함수들을 그대로 재사용한다.

import base64

import fitz  # PyMuPDF
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL, VISION_MODEL

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# 비전 모델에게 페이지 이미지를 해석시킬 때 사용할 지시문
PAGE_PROMPT = """다음은 PDF 문서의 한 페이지 이미지입니다.
페이지에 보이는 내용을 아래 규칙에 따라 텍스트로 변환해주세요.

- 본문은 읽는 순서대로 그대로 전사한다.
- 표는 마크다운 표 형식으로 재구성한다.
- 차트/그림/도표는 "[그림: 핵심 내용 요약]" 형태로 간단히 설명한다.
- 페이지 번호, 머리글/바닥글 등 반복되는 장식 요소는 생략한다.
- 원문 언어를 그대로 유지한다(번역하지 않는다).
- 다른 설명 없이 변환된 텍스트만 출력한다.
"""


def count_pages(pdf_path) -> int:
    """비전 모델 호출 없이 PDF의 전체 페이지 수만 빠르게 센다 (예상 시간 계산용)."""
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    doc.close()
    return n_pages


def extract_page_text(page: fitz.Page) -> str:
    """PyMuPDF로 페이지에 내장된 텍스트 레이어를 직접 추출한다.

    스캔 이미지로만 이루어진 페이지는 텍스트 레이어가 없어 빈 문자열(또는 매우 짧은 문자열)을 반환한다.
    """
    return page.get_text("text").strip()


def page_to_png_bytes(page: fitz.Page, dpi: int) -> bytes:
    """PDF 한 페이지를 지정한 해상도(dpi)의 PNG 이미지 바이트로 렌더링한다."""
    zoom = dpi / 72  # PDF 기본 단위(72dpi) 대비 확대 배율 계산
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix)
    return pixmap.tobytes("png")


def image_to_text(image_bytes: bytes) -> str:
    """페이지 이미지를 gpt-5-mini에 전달해 텍스트로 변환한다."""
    # OpenAI 비전 API는 이미지를 base64 data URL 형태로 전달해야 한다.
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PAGE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content.strip()
