try:
    # Agent components
    from concordia.components import agent as agent_components

    # Associative memory
    from concordia.associative_memory import basic_associative_memory

    # Language model
    from concordia.language_model import language_model

    HAVE_CONCORDIA = True
except Exception as e:
    print("Error:", e)
    HAVE_CONCORDIA = False

print(HAVE_CONCORDIA)
