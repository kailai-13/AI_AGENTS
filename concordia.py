"""
Concordia-compatible Buyer Agent using Llama-3-8B (Ollama)

Save as: concordia_buyer_llama38.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import subprocess
import random
import time

# ============================================================
# Graceful imports (Concordia fall-backs if package not present)
# ============================================================
try:
    from concordia.components import agent as agent_components
    from concordia.associative_memory import associative_memory
    from concordia.language_model import language_model

    HAVE_CONCORDIA = True
except Exception:  # noqa: BLE001
    HAVE_CONCORDIA = False

    class agent_components:  # type: ignore
        class ContextComponent:  # minimal stub
            def make_pre_act_value(self) -> str:
                return ""

            def get_state(self):
                return {}

            def set_state(self, _s):
                pass

    class associative_memory:  # type: ignore
        class AssociativeMemory:
            def __init__(self):
                self._buf: List[Dict[str, Any]] = []

            def add_observation(self, obj: Dict[str, Any]):
                self._buf.append(obj)

            def retrieve(self, k: int = 5):
                return self._buf[-k:]

    class language_model:  # type: ignore
        class LanguageModel:  # echo stub
            def complete(self, prompt: str) -> str:
                return prompt

# ============================================================
# Domain / State
# ============================================================
@dataclass
class Product:
    name: str
    category: str
    quantity: int
    quality_grade: str
    origin: str
    base_market_price: int
    attributes: Dict[str, Any] = field(default_factory=dict)


class DealStatus(str, Enum):
    ONGOING = "ongoing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class NegotiationState:
    round: int = 0
    max_rounds: int = 10
    buyer_budget: int = 0
    seller_min: int = 0
    last_buyer_offer: Optional[int] = None
    last_seller_offer: Optional[int] = None
    history: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================
# Settings
# ============================================================
@dataclass
class Settings:
    buyer_open_pct: float = 0.70
    buyer_accept_pct_market: float = 0.85
    min_concession_abs: int = 1_000
    time_pressure_boost: float = 0.03
    late_round_threshold: int = 4
    close_gap_pct: float = 0.02
    show_logs: bool = True


CFG = Settings()

# ============================================================
# LLM wrapper (Ollama)
# ============================================================
class OllamaLLM(language_model.LanguageModel):  # type: ignore[misc]
    """Thin wrapper around the `ollama` CLI."""

    def __init__(self, model_name: str = "llama-3-8b", timeout: int = 30):
        super().__init__()
        self.model_name = model_name
        self.timeout = timeout

    def complete(self, prompt: str) -> str:  # noqa: D401
        """Send `prompt` to Ollama and return the model's reply."""
        try:
            proc = subprocess.run(
                ["ollama", "run", self.model_name, "--stdin"],
                input=prompt,
                text=True,
                encoding="utf-8",        # ←—— guarantees UTF-8 on Windows
                capture_output=True,
                timeout=self.timeout,
            )
            out = proc.stdout.strip() or proc.stderr.strip()
            return out
        except FileNotFoundError:
            return f"(ollama-missing) {prompt[:400]}"
        except subprocess.TimeoutExpired:
            return "(ollama-timeout) The model did not respond in time."


# ============================================================
# Memory & Personality
# ============================================================
class BuyerMemory(associative_memory.AssociativeMemory):  # type: ignore[misc]
    pass


class BuyerPersonality(agent_components.ContextComponent):  # type: ignore[misc]
    def __init__(
        self,
        persona: str = "analytical-diplomatic",
        traits: Optional[List[str]] = None,
        catchphrases: Optional[List[str]] = None,
    ):
        super().__init__()
        self.persona = persona
        self.traits = traits or ["calm", "data-driven", "fair-but-firm"]
        self.catchphrases = catchphrases or [
            "Let's be fair.",
            "I've done my research.",
            "We can find a middle ground.",
        ]

    def make_pre_act_value(self) -> str:
        return (
            f"You are a buyer agent. Persona: {self.persona}. "
            f"Traits: {', '.join(self.traits)}.\n"
            "Communication style: concise (1–2 sentences), professional, politely firm. "
            f"Use catchphrases sparingly: {', '.join(self.catchphrases)}.\n"
            "Important constraints: NEVER exceed your budget. Prefer win-win outcomes. "
            "If closing conditions are met, accept politely.\n"
        )

    # get_state / set_state inherited from ContextComponent


# ============================================================
# Policy: numeric decision logic only
# ============================================================
class BuyerPolicy:
    """Deterministic numeric negotiation strategy (no LLM involvement)."""

    def opening_offer(self, market_price: int, budget: int) -> int:
        return min(int(market_price * CFG.buyer_open_pct), budget)

    def should_accept(self, seller_price: int, market_price: int, budget: int) -> bool:
        return (seller_price <= budget) and (
            seller_price <= int(market_price * CFG.buyer_accept_pct_market)
        )

    def counter_offer(
        self,
        seller_price: int,
        market_price: int,
        budget: int,
        last_buyer_offer: Optional[int],
        round_i: int,
        max_rounds: int,
    ) -> int:
        last = last_buyer_offer or self.opening_offer(market_price, budget)
        rounds_left = max(1, max_rounds - round_i)

        # dynamic concession
        step_pct = CFG.time_pressure_boost / rounds_left
        step = max(CFG.min_concession_abs, int(step_pct * market_price))

        target = (
            int(last + 0.25 * (seller_price - last))
            if rounds_left > CFG.late_round_threshold
            else int(last + 0.5 * (seller_price - last))
        )
        counter = min(target + step, budget)

        # close tiny gaps
        if abs(seller_price - counter) <= int(CFG.close_gap_pct * market_price):
            counter = min(seller_price, budget)

        return max(counter, last)  # never go backwards


# ============================================================
# Concordia Agent
# ============================================================
class ConcordiaBuyerAgent(
    agent_components.ContextComponent if HAVE_CONCORDIA else object  # type: ignore[misc]
):
    """Buyer agent that merges numeric policy with LLM phrasing."""

    def __init__(self, name: str = "BuyerAgent", model_name: str = "llama-3-8b"):
        self.name = name
        self.personality = BuyerPersonality()
        self.memory = BuyerMemory()
        self.model = OllamaLLM(model_name=model_name)
        self.policy = BuyerPolicy()

    # --------------------------------------------------------
    # Main action loop
    # --------------------------------------------------------
    def act(self, observation: Dict[str, Any], state: NegotiationState) -> Dict[str, Any]:
        product = observation.get("product", {})
        market = int(product.get("base_market_price", 0))
        seller_offer = observation.get("seller_offer")

        # ========== Opening ==========
        if seller_offer is None:
            offer = self.policy.opening_offer(market, state.buyer_budget)
            text = self._llm_opening(product.get("name", "item"), market, offer)
            self.memory.add_observation(
                {"round": state.round, "role": "buyer", "offer": offer, "text": text}
            )
            return {"text": text, "offer": offer, "status": DealStatus.ONGOING}

        seller_price = int(seller_offer)

        # ========== Accept? ==========
        if self.policy.should_accept(seller_price, market, state.buyer_budget):
            text = self._llm_accept(product.get("name", "item"), market, seller_price, state)
            self.memory.add_observation(
                {
                    "round": state.round,
                    "role": "buyer",
                    "offer": seller_price,
                    "text": text,
                    "accepted": True,
                }
            )
            return {"text": text, "offer": seller_price, "status": DealStatus.ACCEPTED}

        # ========== Counter ==========
        counter = self.policy.counter_offer(
            seller_price,
            market,
            state.buyer_budget,
            state.last_buyer_offer,
            state.round,
            state.max_rounds,
        )
        text = self._llm_counter(product.get("name", "item"), market, seller_price, counter, state)
        self.memory.add_observation(
            {"round": state.round, "role": "buyer", "offer": counter, "text": text}
        )
        return {"text": text, "offer": counter, "status": DealStatus.ONGOING}

    # --------------------------------------------------------
    # LLM helpers
    # --------------------------------------------------------
    def _llm_opening(self, item: str, market: int, offer: int) -> str:
        prompt = (
            f"{self.personality.make_pre_act_value()}\n"
            f"Situation: Opening offer for {item} (market ₹{market}).\n"
            f"Your offer (numeric): ₹{offer}.\n"
            "Write a 1–2 sentence buyer message in the persona above, concise and professional. "
            "Start the message with the offer amount and keep the tone consistent.\n"
        )
        return self.model.complete(prompt)

    def _llm_accept(self, item: str, market: int, seller_price: int, state: NegotiationState) -> str:
        prompt = (
            f"{self.personality.make_pre_act_value()}\n"
            f"Situation: Seller offered ₹{seller_price} for {item}. "
            f"Your budget: ₹{state.buyer_budget}. Market: ₹{market}.\n"
            "Write a single-sentence polite acceptance (e.g., 'Deal at ₹X. Thank you.').\n"
        )
        return self.model.complete(prompt)

    def _llm_counter(
        self,
        item: str,
        market: int,
        seller_price: int,
        counter: int,
        state: NegotiationState,
    ) -> str:
        prompt = (
            f"{self.personality.make_pre_act_value()}\n"
            f"Situation: Seller offered ₹{seller_price} for {item} (market ₹{market}).\n"
            f"Your numeric decision: counter at ₹{counter}. "
            f"Budget: ₹{state.buyer_budget}. "
            f"Round {state.round}/{state.max_rounds}.\n"
            "Constraints: 1) Do NOT exceed the numeric decision. "
            "2) Keep it to 1–2 sentences, professional, optional catchphrase. "
            "3) Give one persuasive reason (market data, quality, partnership).\n"
            "Write the buyer message now.\n"
        )
        return self.model.complete(prompt)


# ============================================================
# Factory (what external runners import)
# ============================================================
def make_buyer_agent(config: Optional[Dict[str, Any]] = None) -> ConcordiaBuyerAgent:
    cfg = config or {}
    return ConcordiaBuyerAgent(
        name=cfg.get("name", "ConcordiaBuyer"),
        model_name=cfg.get("model", "llama-3-8b"),
    )


# ============================================================
# Quick local smoke test (runs without Concordia)
# ============================================================
def _local_demo() -> None:
    product = {"name": "Alphonso Mangoes", "base_market_price": 180_000}
    state = NegotiationState(
        round=1,
        max_rounds=10,
        buyer_budget=200_000,
        seller_min=150_000,
    )
    buyer = ConcordiaBuyerAgent("DemoBuyer")

    seller_price = int(product["base_market_price"] * 1.5)
    print(f"R1 Seller asks ₹{seller_price}")
    out = buyer.act({"product": product, "seller_offer": seller_price}, state)
    print(f"R1 Buyer offers ₹{out['offer']} :: {out['text'][:120]}")

    for r in range(2, 8):
        state.round = r
        seller_price = max(state.seller_min, int((seller_price + out["offer"]) / 2))
        print(f"R{r} Seller asks ₹{seller_price}")
        out = buyer.act({"product": product, "seller_offer": seller_price}, state)
        print(f"R{r} Buyer offers ₹{out['offer']} :: {out['text'][:120]}")
        if out["status"] == DealStatus.ACCEPTED or out["offer"] >= seller_price:
            print("Deal reached.")
            return
    print("Demo ended without deal.")


if __name__ == "__main__":
    print("\nRunning local demo of ConcordiaBuyerAgent …\n")
    _local_demo()
