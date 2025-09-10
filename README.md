 
---

# 🥭 AI Agents – Concordia Buyer Negotiation 

## 📌 Overview

This project implements an **AI Buyer Agent** for automated negotiations using the **Concordia framework** (if available) and a fallback mode for standalone testing.
It integrates a **personality-driven negotiation policy** with **numeric constraints** and optional **Ollama LLaMA 3 8B** local LLM support for generating natural language negotiation responses.

---

## 📂 Project Structure 

```
AI_AGENTS/
│
├── __pycache__/               # Python bytecode cache
├── .python-version            # Python version lock file
├── concordia.py               # Unified Buyer Agent (Concordia-compatible)
├── interview_negotiation.py   # Example/template for multi-round negotiation
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
└── test.py                    # Test cases (easy, medium, hard negotiation scenarios)
```

---

## ⚙ Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Buyer Agent Simulation

Run a single negotiation test:

```bash
python concordia.py
```

Run multiple difficulty tests:

```bash
python test.py
```

---

## 🧪 Test Scenarios

The **`test.py`** file contains:

* **Easy** – Cooperative seller, quick agreement
* **Medium** – Balanced back-and-forth
* **Hard** – Aggressive seller, late concessions
* **Very Hard** – Tight budget vs high seller price
* **Randomized** – Random market price & budget
* **Extreme** – Almost no budget flexibility

---

## 🤖 LLM Integration

* Uses **Ollama** for local LLaMA 3.8B inference
* Falls back to echo mode if Ollama is not installed

To use with Ollama:

```bash
ollama pull llama3:8b
```

---

## 📜 License

MIT License – free to use, modify, and distribute.

---
