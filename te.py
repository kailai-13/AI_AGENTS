try:
    from concordia.components import agent as agent_components
    from concordia.associative_memory import associative_memory
    from concordia.language_model import language_model
    HAVE_CONCORDIA = True
except Exception:  # Concordia not installed
    HAVE_CONCORDIA = False

print(HAVE_CONCORDIA)