def truncate(s, max_length=500) -> str:
    s = str(s)
    if len(s) <= max_length:
        return s
    return s[:max_length] + "..."
