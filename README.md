
---



```markdown
# Concordia Unified Buyer Agent 🤝

An intelligent buyer agent for negotiation simulations, compatible with **Concordia** and tested with **Ollama's LLaMA 3.8B** model.  
It uses a mix of **data-driven negotiation policy** (math rules) + **LLM-powered personality messages** to make effective, budget-safe offers.

---

## ✨ Features
- Concordia-compatible `make_buyer_agent()` factory.
- Pure numeric negotiation logic for consistent price decisions.
- LLaMA-based message generation via [Ollama](https://ollama.com/).
- Personality-driven buyer with memory of recent conversation turns.
- Flexible — can run in Concordia framework or standalone mode.
- Multi-scenario testing with **Easy / Medium / Hard** difficulty.

---

## 📂 Project Structure
```

.
├── unified\_buyer\_agent.py      # Main buyer agent class & negotiation logic
├── test\_unified\_buyer\_multi.py # Test runner with 6 negotiation scenarios
├── requirements.txt            # Python dependencies
└── README.md                   # This file

````

---

## ⚙ Requirements

1. **Python 3.9+**
2. [Ollama](https://ollama.com/) installed locally with:
   ```bash
   ollama pull llama3:8b
````

3. Concordia (optional — the code can run without it, with fallbacks).

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/concordia-buyer-agent.git
cd concordia-buyer-agent

# Install dependencies
pip install -r requirements.txt
```

---

## ▶ Running Tests

Run all 6 negotiation difficulty scenarios:

```bash
python test.py
```

---

## 🛠 Example Output

```
================================================================================
TEST SCENARIO: Easy 1
Product: Local Bananas | Market ₹50,000 | Budget ₹60,000
================================================================================

SELLER: Asking ₹55,000 (reasonable price)
BUYER OFFER: ₹35,000 | Let's be fair, I believe ₹35,000 is a reasonable starting point.

SELLER: Counter offer: ₹50,000
BUYER: ₹50,000 | Deal accepted at ₹50,000. I’ve done my research.

🎉 DEAL ACCEPTED!
Final Price: ₹50,000 | Savings: ₹10,000
```

---

## 🧠 How It Works

* **Numeric Policy** (`_Policy` class) decides *when* to accept and *how much* to counter.
* **Personality + Memory** shape *how* to say it, using LLaMA-generated messages.
* **Fallback Mode**: Runs without Concordia if not installed.

---

## 📜 License

MIT License — feel free to use and modify.

---

## 💡 Credits

* [Concordia Framework](https://github.com/concordia-agents/concordia)
* [Ollama](https://ollama.com/) for local LLM serving

````

---

## **`requirements.txt`**
```txt
# Core dependencies
dataclasses; python_version < '3.10'
typing-extensions
enum34; python_version < '3.4'

# Optional but recommended
concordia==0.1.0  # Remove or change version if unavailable

# Local LLM integration
ollama  # Python client for Ollama (if available)

# Dev & testing
pytest
````

---

