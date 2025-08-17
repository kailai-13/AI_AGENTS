try:
    import concordia
    HAVE_CONCORDIA = True
except Exception:  # Concordia not installed
    HAVE_CONCORDIA = False

print(HAVE_CONCORDIA)