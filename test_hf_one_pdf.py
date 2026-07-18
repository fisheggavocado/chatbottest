# HF Dataset의 PDF 1개만 시험 처리하는 스크립트 (실제 상태를 건드리지 않는 격리 테스트)
#
# 기존 main.py와의 차이:
#   - 인덱스/checkpoint를 실제 OUTPUT_DIR 대신 output_test/ 에 저장한다.
#   - 처리 완료 후 HF 업로드를 하지 않는다 (HF repo의 embedding/ 폴더 오염 방지).
#   - 실제 output/ 과 checkpoint.json은 전혀 건드리지 않으므로,
#     테스트 후 본 실행(python main.py --use-hf)은 처음부터 정상 진행된다.
#
# 사용법 (프로젝트 폴더에서 실행):
#   python test_hf_one_pdf.py            # HF에서 동기화한 PDF 중 1개만 처리 -> output_test/
#   python test_hf_one_pdf.py --limit 2  # 2개 처리
#   python test_hf_one_pdf.py --clean    # 원상복구: output_test/ 와 hf_pdfs/ 캐시 삭제

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
TEST_OUTPUT_DIR = PROJECT_DIR / "output_test"
HF_SYNC_DIR = PROJECT_DIR / "hf_pdfs"  # config.HF_LOCAL_SYNC_DIR("./hf_pdfs")과 동일 위치


def clean() -> None:
    """테스트 산출물을 모두 삭제해 원래 상태로 되돌린다."""
    for target in (TEST_OUTPUT_DIR, HF_SYNC_DIR):
        if target.exists():
            shutil.rmtree(target)
            print(f"[삭제 완료] {target}")
        else:
            print(f"[이미 없음] {target}")
    print("[원상복구 완료] 실제 output/ 폴더와 HF repo는 처음부터 건드리지 않았습니다.")


def run(limit: int) -> None:
    # 무거운 모듈(main -> BGE-M3 로딩)을 import 하기 전에 config의 출력 경로를 먼저 바꿔치기한다.
    import config

    config.OUTPUT_DIR = str(TEST_OUTPUT_DIR)

    import checkpoint

    # checkpoint.py는 import 시점에 경로를 계산하므로, 테스트 폴더를 보도록 다시 지정한다.
    checkpoint.CHECKPOINT_PATH = TEST_OUTPUT_DIR / "checkpoint.json"

    import main as pipeline

    pipeline.OUTPUT_DIR = str(TEST_OUTPUT_DIR)
    # main.py 마지막의 upload_output_to_hf() 호출을 막는다. (HF에서 내려받는 sync는 정상 동작)
    pipeline.HF_REPO_ID = None

    print(f"[테스트 모드] 결과 저장 위치: {TEST_OUTPUT_DIR} (실제 output/은 사용하지 않음)")
    print("[테스트 모드] 처리 완료 후 HF 업로드는 생략됩니다.")

    sys.argv = ["main.py", "--use-hf", "--limit", str(limit)]
    pipeline.main()

    print()
    print("[테스트 완료] 결과 확인 후 원래대로 되돌리려면: python test_hf_one_pdf.py --clean")


def cli() -> None:
    parser = argparse.ArgumentParser(description="HF Dataset PDF 1개 격리 테스트 (실제 output/ 미사용)")
    parser.add_argument("--limit", type=int, default=1, help="처리할 PDF 개수 (기본 1)")
    parser.add_argument("--clean", action="store_true", help="테스트 산출물(output_test/, hf_pdfs/) 삭제")
    args = parser.parse_args()

    if args.clean:
        clean()
    else:
        run(args.limit)


if __name__ == "__main__":
    cli()
