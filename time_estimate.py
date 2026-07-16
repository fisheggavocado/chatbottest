# 인덱싱 시작 전, PDF 총 페이지 수를 기준으로 예상 소요 시간을 안내하는 모듈
# 실제 API를 호출하기 전이라 정확한 값은 알 수 없으므로, 두 단계로 안내한다:
#   1) print_upfront_estimate() — 설정된 가정치(SECONDS_PER_PAGE_ESTIMATE)로 대략치를 먼저 보여줌
#   2) print_measured_estimate() — 첫 페이지를 실제 처리한 뒤, 실측 시간으로 재추정

from config import SECONDS_PER_PAGE_ESTIMATE


def format_duration(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}시간 {minutes}분"
    if minutes:
        return f"{minutes}분 {sec}초"
    return f"{sec}초"


def print_upfront_estimate(total_pages: int) -> None:
    """실제 처리 전, 설정된 가정치로 대략적인 예상 시간을 안내한다."""
    if total_pages == 0:
        return
    estimated_seconds = total_pages * SECONDS_PER_PAGE_ESTIMATE
    print(
        f"[예상 시간 · 가정치] {total_pages}페이지 x {SECONDS_PER_PAGE_ESTIMATE}초/페이지 "
        f"≈ {format_duration(estimated_seconds)} (실측 아님, 첫 페이지 처리 후 재추정됩니다)"
    )


def print_measured_estimate(seconds_per_page: float, remaining_pages: int) -> None:
    """첫 페이지의 실제 처리 시간을 기준으로 남은 시간을 재추정한다."""
    estimated_seconds = seconds_per_page * remaining_pages
    print(
        f"[예상 시간 · 실측 기반] 페이지당 실제 {seconds_per_page:.1f}초 x 남은 {remaining_pages}페이지 "
        f"≈ {format_duration(estimated_seconds)} 남음"
    )
