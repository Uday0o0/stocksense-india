from datetime import datetime
from zoneinfo import ZoneInfo


def is_market_open(now=None):
    current_time = now or datetime.now(ZoneInfo("Asia/Kolkata"))
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
    else:
        current_time = current_time.astimezone(ZoneInfo("Asia/Kolkata"))

    minutes = current_time.hour * 60 + current_time.minute
    return current_time.weekday() < 5 and 555 <= minutes <= 930
