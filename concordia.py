"""
Unified Buyer Agent
-------------------
• Concordia-compatible  (make_buyer_agent)
• Works with custom BaseBuyerAgent evaluation template
• Personality-driven, budget-safe, memory-aware
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import subprocess
import re

# ============================================================
# Graceful Concordia imports ─ fall back to local stubs
# ============================================================
try:
    from concordia.components import agent as agent_components
    from concordia.associative_memory import associative_memory
    from concordia.language_model import language_model

    HAVE_CONCORDIA = True
except Exception:  # Concordia not installed
    HAVE_CONCORDIA = False

    class agent_components:  # type: ignore
        class ContextComponent:
            def make_pre_act_value(self) -> str: return ""
            def get_state(self): return {}
            def set_state(self, _): pass

    class associative_memory:  # type: ignore
        class AssociativeMemory:
            def __init__(self): self._buf: List[Dict[str, Any]] = []
            def add_observation(self, obj): self._buf.append(obj)
            def retrieve(self, k: int = 5): return self._buf[-k:]

    class language_model:  # type: ignore
        class LanguageModel:
            def complete(self, prompt: str) -> str: return prompt  # echo


# ============================================================
# ---------- Part 1: Common dataclasses & enums --------------
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


@dataclass
class NegotiationContext:
    product: Product
    your_budget: int
    current_round: int
    seller_offers: List[int]
    your_offers: List[int]
    messages: List[Dict[str, str]]


class DealStatus(str, Enum):
    ONGOING = "ongoing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


# ============================================================
# ---------- Part 2: Numeric negotiation policy --------------
# ============================================================
@dataclass
class _Cfg:
    open_anchor_pct: float = 0.70
    accept_pct_market: float = 0.85
    min_step: int = 1_000
    time_boost_pct: float = 0.03
    late_round: int = 4
    close_gap_pct: float = 0.02


CFG = _Cfg()


class _Policy:
    """Pure maths — no LLM necessary."""

    @staticmethod
    def opening_offer(market: int, budget: int) -> int:
        anchor = int(market * CFG.open_anchor_pct)
        # If budget is tighter than anchor, open at 90 % of budget (still leaves room)
        return min(anchor, max(int(budget * 0.9), 1_000))

    @staticmethod
    def should_accept(price: int, market: int, budget: int) -> bool:
        return price <= budget and price <= int(market * CFG.accept_pct_market)

    @staticmethod
    def counter(
        seller_price: int,
        market: int,
        budget: int,
        last_offer: Optional[int],
        round_i: int,
        max_rounds: int,
    ) -> int:
        last = last_offer or _Policy.opening_offer(market, budget)
        rounds_left = max(1, max_rounds - round_i)

        step_pct = CFG.time_boost_pct / rounds_left
        step = max(CFG.min_step, int(step_pct * market))

        target = (
            int(last + 0.25 * (seller_price - last))
            if rounds_left > CFG.late_round
            else int(last + 0.5 * (seller_price - last))
        )
        offer = min(target + step, budget)

        if abs(seller_price - offer) <= int(CFG.close_gap_pct * market):
            offer = min(seller_price, budget)

        return max(offer, last)


# ============================================================
# ---------- Part 3: Personality & Memory --------------------
# ============================================================
class _Memory(associative_memory.AssociativeMemory):  # type: ignore[misc]
    """Wraps Concordia memory; stores full conversation lines."""

    def last_turns(self, k: int = 3) -> str:
        buf = self.retrieve(k)
        return "\n".join(f"{b['role'].capitalize()}: {b['text']}" for b in buf)


class _Personality(agent_components.ContextComponent):  # type: ignore[misc]
    def __init__(self):
        self.type = "data-driven diplomat"
        self.traits = ["calm", "analytical", "relationship-oriented"]
        self.catchphrases = ["Let's be fair.", "I've done my research."]
        super().__init__()

    # -------- Context for every LLM prompt ----------
    def make_pre_act_value(self) -> str:
        traits = ", ".join(self.traits)
        phrases = ", ".join(self.catchphrases)
        return (
            f"You are a buyer who is {self.type}. Traits: {traits}. "
            "Speak in 1–2 concise, professional sentences; optionally use one catchphrase "
            f"from: {phrases}. Never exceed the numeric decision you are given.\n"
        )


# ============================================================
# ---------- Part 4: LLM wrapper (Ollama) --------------------
# ============================================================
class _OllamaLLM(language_model.LanguageModel):  # type: ignore[misc]
    """UTF-8 safe wrapper (prompt via stdin)."""

    def __init__(self, model: str = "llama3:8b", timeout: int = 60):
        super().__init__()
        self.model, self.timeout = model, timeout

    def complete(self, prompt: str) -> str:
        try:
            proc = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=self.timeout,
            )
            return (proc.stdout or proc.stderr).strip()
        except Exception:
            # Fallback echo for environments without Ollama
            return f"(echo) {prompt.splitlines()[-1][:80]}"


# ============================================================
# ---------- Part 5: Unified Buyer Agent ---------------------
# ============================================================
class UnifiedBuyerAgent(
    agent_components.ContextComponent if HAVE_CONCORDIA else object  # Concordia
):
    """
    One class = one brain.
    Exposes both Concordia `.act()` and template `.generate_* / respond_*` APIs.
    """

    def __init__(self, name: str = "Buyer", model_name: str = "llama3:8b"):
        self.name = name
        self.personality = _Personality()
        self.memory = _Memory()
        self.policy = _Policy()
        self.llm = _OllamaLLM(model_name)

    # =====================================================
    # --- 5A.  Concordia interface (observation → action) --
    # =====================================================
    def act(self, observation: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Concordia runner expects this."""
        product = observation["product"]
        market = int(product["base_market_price"])
        budget = state["buyer_budget"]
        round_i = state["round"]
        max_rounds = state["max_rounds"]
        seller_offer = observation.get("seller_offer")

        if seller_offer is None:  # opening
            my_offer = self.policy.opening_offer(market, budget)
            txt = self._msg_open(product["name"], market, my_offer)
            status = DealStatus.ONGOING
        else:
            seller_price = int(seller_offer)
            if self.policy.should_accept(seller_price, market, budget):
                my_offer = seller_price
                txt = self._msg_accept(seller_price)
                status = DealStatus.ACCEPTED
            else:
                my_offer = self.policy.counter(
                    seller_price, market, budget, state.get("last_buyer_offer"), round_i, max_rounds
                )
                txt = self._msg_counter(seller_price, my_offer, product["name"], round_i, max_rounds)
                status = DealStatus.ONGOING

        # Memorise
        self.memory.add_observation({"role": "buyer", "text": txt})
        return {"text": txt, "offer": my_offer, "status": status}

    # =====================================================
    # --- 5B.  Template BaseBuyerAgent API ----------------
    # =====================================================
    # These six methods let the same class satisfy the interview template.

    # 1. Personality dict
    def define_personality(self) -> Dict[str, Any]:
        return {
            "personality_type": self.personality.type,
            "traits": self.personality.traits,
            "negotiation_style": (
                "Quotes data, remains calm, aims for win-win, concedes faster near deadline."
            ),
            "catchphrases": self.personality.catchphrases,
        }

    # 2. Opening offer
    def generate_opening_offer(self, ctx: NegotiationContext) -> Tuple[int, str]:
        offer = self.policy.opening_offer(ctx.product.base_market_price, ctx.your_budget)
        msg = self._msg_open(ctx.product.name, ctx.product.base_market_price, offer)
        self.memory.add_observation({"role": "buyer", "text": msg})
        return offer, msg

    # 3. Respond to seller
    def respond_to_seller_offer(  # noqa: D401
        self, ctx: NegotiationContext, seller_price: int, seller_msg: str
    ) -> Tuple[DealStatus, int, str]:
        # Store seller line
        self.memory.add_observation({"role": "seller", "text": seller_msg})

        market = ctx.product.base_market_price
        budget = ctx.your_budget

        if self.policy.should_accept(seller_price, market, budget):
            msg = self._msg_accept(seller_price)
            self.memory.add_observation({"role": "buyer", "text": msg})
            return DealStatus.ACCEPTED, seller_price, msg

        offer = self.policy.counter(
            seller_price, market, budget, ctx.your_offers[-1] if ctx.your_offers else None, ctx.current_round, 10
        )
        msg = self._msg_counter(seller_price, offer, ctx.product.name, ctx.current_round, 10)
        self.memory.add_observation({"role": "buyer", "text": msg})
        return DealStatus.ONGOING, offer, msg

    # 4. Personality prompt (for external evaluation)
    def get_personality_prompt(self) -> str:
        return self.personality.make_pre_act_value()

    # -----------------------------------------------------
    # LLM helper texts
    # -----------------------------------------------------
    def _msg_open(self, item: str, market: int, offer: int) -> str:
        prompt = (
            self.personality.make_pre_act_value()
            + f"Context: Opening offer for {item}. Market ₹{market}. Your numeric offer: ₹{offer}.\n"
            "Compose the buyer message."
        )
        return self.llm.complete(prompt)

    def _msg_accept(self, price: int) -> str:
        prompt = (
            self.personality.make_pre_act_value()
            + f"Context: You are accepting the seller's offer of ₹{price}. Reply with one concise sentence."
        )
        return self.llm.complete(prompt)

    def _msg_counter(
        self, seller_price: int, counter: int, item: str, round_i: int, max_rounds: int
    ) -> str:
        recent = self.memory.last_turns(2)
        urgency = "final" if max_rounds - round_i <= CFG.late_round else "normal"
        prompt = (
            self.personality.make_pre_act_value()
            + f"Previous lines:\n{recent}\n\n"
            f"Seller price: ₹{seller_price}. Your counter decision: ₹{counter}. "
            f"Round {round_i}/{max_rounds} ({urgency}). "
            "Write a persuasive 1–2 sentence counter without exceeding the numeric offer."
        )
        return self.llm.complete(prompt)


# ============================================================
# ---------- Part 6: Factory & local demo --------------------
# ============================================================
def make_buyer_agent(cfg: Optional[Dict[str, Any]] = None) -> UnifiedBuyerAgent:
    cfg = cfg or {}
    return UnifiedBuyerAgent(
        name=cfg.get("name", "ConcordiaBuyer"),
        model_name=cfg.get("model", "llama3:8b"),
    )


# ---------------- Local demo for template harness -----------
if __name__ == "__main__":
    # Quick smoke test with the template’s MockSellerAgent
    from random import randint

    product = Product(
        name="Alphonso Mangoes",
        category="Mangoes",
        quantity=100,
        quality_grade="A",
        origin="Ratnagiri",
        base_market_price=180_000,
    )
    budget = 200_000
    # Minimal evaluation loop
    ctx = NegotiationContext(product, budget, 0, [], [], [])
    buyer = UnifiedBuyerAgent()
    from pprint import pprint

    # Seller mock
    seller_price = int(product.base_market_price * 1.5)
    seller_msg = f"Premium fruit, asking ₹{seller_price}"
    ctx.seller_offers.append(seller_price)
    ctx.messages.append({"role": "seller", "message": seller_msg})

    offer, msg = buyer.generate_opening_offer(ctx)
    ctx.your_offers.append(offer)
    ctx.messages.append({"role": "buyer", "message": msg})
    print(f"R1 Buyer: ₹{offer} | {msg[:120]}")

    # simple loop
    for r in range(2, 10):
        ctx.current_round = r
        # naive seller concession
        seller_price = max(int(product.base_market_price * 0.8), int((seller_price + offer) / 2))
        seller_msg = f"Counter ₹{seller_price}"
        status, offer, msg = buyer.respond_to_seller_offer(ctx, seller_price, seller_msg)
        ctx.your_offers.append(offer)
        ctx.messages.append({"role": "buyer", "message": msg})
        print(f"R{r} Buyer: ₹{offer} | {msg[:120]}")
        if status == DealStatus.ACCEPTED or offer >= seller_price:
            print("Deal closed.")
            break
