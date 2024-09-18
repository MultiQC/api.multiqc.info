def strtobool(val) -> bool:
    return str(val).lower() in ("y", "yes", "t", "true", "on", "1")
