"""
===========================================
AI NEGOTIATION AGENT - CONCORDIA ENHANCED
===========================================

Buyer agent that integrates Concordia framework with the competition template structure.
Features personality-driven negotiations, memory-aware decisions, and LLM-generated responses.
"""

import json
import re
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import random

# ============================================================
# Graceful Concordia imports ─ fall back to local stubs
# ============================================================
try:
    from concordia.components import agent as agent_components
    from concordia.associative_memory import basic_associative_memory
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

# ============================================
# PART 1: DATA STRUCTURES (DO NOT MODIFY)
# ============================================

@dataclass
class Product:
    """Product being negotiated"""
    name: str
    category: str
    quantity: int
    quality_grade: str  # 'A', 'B', or 'Export'
    origin: str
    base_market_price: int  # Reference price for this product
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NegotiationContext:
    """Current negotiation state"""
    product: Product
    your_budget: int  # Your maximum budget (NEVER exceed this)
    current_round: int
    seller_offers: List[int]  # History of seller's offers
    your_offers: List[int]  # History of your offers
    messages: List[Dict[str, str]]  # Full conversation history

class DealStatus(Enum):
    ONGOING = "ongoing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"

# ============================================
# PART 2: BASE AGENT CLASS (DO NOT MODIFY)
# ============================================

class BaseBuyerAgent(ABC):
    """Base class for all buyer agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.personality = self.define_personality()
        
    @abstractmethod
    def define_personality(self) -> Dict[str, Any]:
        """Define your agent's personality traits."""
        pass
    
    @abstractmethod
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        """Generate your first offer in the negotiation."""
        pass
    
    @abstractmethod
    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        """Respond to the seller's offer."""
        pass
    
    @abstractmethod
    def get_personality_prompt(self) -> str:
        """Return a prompt that describes how your agent should communicate."""
        pass

# ============================================================
# CONCORDIA COMPONENTS
# ============================================================

@dataclass
class _PolicyConfig:
    """Configuration for negotiation policy"""
    open_anchor_pct: float = 0.70
    accept_pct_market: float = 0.85
    min_step: int = 1_000
    time_boost_pct: float = 0.03
    late_round: int = 4
    close_gap_pct: float = 0.02

CFG = _PolicyConfig()

class _NegotiationPolicy:
    """Mathematical negotiation strategy - no LLM needed"""

    @staticmethod
    def opening_offer(market: int, budget: int) -> int:
        anchor = int(market * CFG.open_anchor_pct)
        # If budget is tighter than anchor, open at 90% of budget (still leaves room)
        return min(anchor, max(int(budget * 0.9), 1_000))

    @staticmethod
    def should_accept(price: int, market: int, budget: int) -> bool:
        return price <= budget and price <= int(market * CFG.accept_pct_market)

    @staticmethod
    def counter_offer(
        seller_price: int,
        market: int,
        budget: int,
        last_offer: Optional[int],
        round_i: int,
        max_rounds: int,
    ) -> int:
        last = last_offer or _NegotiationPolicy.opening_offer(market, budget)
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

class _ConcordiaMemory(basic_associative_memory.AssociativeMemoryBank):  # type: ignore[misc]
 # type: ignore[misc]
    """Enhanced memory that stores full conversation context"""

    def __init__(self):
        super().__init__()
        self._conversation_buffer: List[Dict[str, Any]] = []

    def add_conversation_turn(self, role: str, message: str, price: Optional[int] = None):
        """Add a conversation turn to memory"""
        entry = {"role": role, "text": message}
        if price is not None:
            entry["price"] = price
        self._conversation_buffer.append(entry)
        self.add_observation(entry)

    def get_recent_turns(self, k: int = 3) -> str:
        """Get recent conversation turns as formatted string"""
        recent = self._conversation_buffer[-k:] if self._conversation_buffer else []
        return "\n".join(f"{turn['role'].capitalize()}: {turn['text']}" for turn in recent)

    def get_negotiation_summary(self) -> str:
        """Get summary of negotiation progress"""
        if not self._conversation_buffer:
            return "No conversation yet."
        
        turns = len(self._conversation_buffer)
        return f"Conversation has {turns} turns. Recent context available."

class _PersonalityComponent(basic_associative_memory.AssociativeMemoryBank):  # type: ignore[misc]
    """Concordia personality component"""
    
    def __init__(self):
        super().__init__()
        self.type = "analytical-diplomatic"
        self.traits = ["calm", "strategic", "data-driven", "fair"]
        self.catchphrases = [
            "Let's be fair to both sides.",
            "I've done my research.", 
            "We can find a middle ground."
        ]
        self.style = "professional and respectful"

    def make_pre_act_value(self) -> str:
        """Generate personality context for LLM prompts"""
        traits_str = ", ".join(self.traits)
        phrases_str = ", ".join(self.catchphrases)
        return (
            f"You are a {self.type} buyer agent. Your traits: {traits_str}. "
            f"Your communication style is {self.style}. "
            f"Use phrases like: {phrases_str}. "
            "Speak in 1-2 professional, concise sentences. "
            "Never exceed your numeric budget constraints.\n"
        )

class _OllamaLLM(language_model.LanguageModel):
    def __init__(self, model: str = "llama3:8b", timeout: int = 60):
        super().__init__()
        self.model = model
        self.timeout = timeout

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
            response = (proc.stdout or proc.stderr).strip()
            return response if response else "I understand. Let me respond appropriately."
        except Exception:
            return "Based on the context, I'll proceed with the negotiation."

    def sample_choice(self, prompt: str, choices: list) -> str:
        # Example implementation: simple first choice or fallback to complete
        if choices:
            return choices[0]
        else:
            return self.complete(prompt)

    def sample_text(self, prompt: str) -> str:
        # Redirect to complete for text generation
        return self.complete(prompt)


# ============================================
# PART 3: CONCORDIA-ENHANCED BUYER AGENT
# ============================================

class YourBuyerAgent(BaseBuyerAgent):
    """
    Concordia-enhanced buyer agent with personality, memory, and LLM integration
    """
    
    def __init__(self, name: str = "ConcordiaBuyer", model_name: str = "llama3:8b"):
        super().__init__(name)
        
        # Concordia components
        self.personality_component = _PersonalityComponent()
        self.memory = _ConcordiaMemory()
        self.policy = _NegotiationPolicy()
        self.llm = _OllamaLLM(model_name)
        
        # Initialize personality from Concordia component
        self.personality.update({
            "personality_type": self.personality_component.type,
            "traits": self.personality_component.traits,
            "catchphrases": self.personality_component.catchphrases
        })
    
    def define_personality(self) -> Dict[str, Any]:
        """Define personality traits using Concordia component"""
        return {
            "personality_type": "analytical-diplomatic",
            "traits": [
                "Calm under pressure",
                "Strategic thinker", 
                "Fair but firm",
                "Data-driven decisions",
                "Relationship-focused"
            ],
            "negotiation_style": (
                "Uses market data and fairness as leverage. Starts reasonable, "
                "increases offers strategically, accelerates near deadline."
            ),
            "catchphrases": [
                "Let's be fair to both sides.",
                "I've done my research.",
                "We can find a middle ground."
            ]
        }
    
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        """Generate opening offer using Concordia components"""
        # Use policy to determine offer amount
        offer_price = self.policy.opening_offer(
            context.product.base_market_price, 
            context.your_budget
        )
        
        # Generate message using LLM and personality
        message = self._generate_opening_message(context, offer_price)
        
        # Store in memory
        self.memory.add_conversation_turn("buyer", message, offer_price)
        
        return offer_price, message
    
    def respond_to_seller_offer(
        self, context: NegotiationContext, seller_price: int, seller_message: str
    ) -> Tuple[DealStatus, int, str]:
        """Respond to seller using Concordia strategy"""
        
        # Store seller's offer in memory
        self.memory.add_conversation_turn("seller", seller_message, seller_price)
        
        market_price = context.product.base_market_price
        
        # Check if we should accept using policy
        if self.policy.should_accept(seller_price, market_price, context.your_budget):
            message = self._generate_acceptance_message(seller_price)
            self.memory.add_conversation_turn("buyer", message, seller_price)
            return DealStatus.ACCEPTED, seller_price, message
        
        # Generate counter offer using policy
        last_offer = context.your_offers[-1] if context.your_offers else None
        counter_price = self.policy.counter_offer(
            seller_price, market_price, context.your_budget,
            last_offer, context.current_round, 10
        )
        
        # Generate counter message using LLM
        message = self._generate_counter_message(
            context, seller_price, counter_price, seller_message
        )
        
        # Store in memory
        self.memory.add_conversation_turn("buyer", message, counter_price)
        
        return DealStatus.ONGOING, counter_price, message
    
    def get_personality_prompt(self) -> str:
        """Get personality prompt from Concordia component"""
        return self.personality_component.make_pre_act_value()
    
    def _generate_opening_message(self, context: NegotiationContext, offer_price: int) -> str:
        """Generate opening message using LLM"""
        product = context.product
        market = product.base_market_price
        
        prompt = (
            self.personality_component.make_pre_act_value() +
            f"Context: Opening negotiation for {product.name} ({product.quality_grade}-grade, "
            f"{product.quantity}kg from {product.origin}). "
            f"Market price: ₹{market:,}. Your opening offer: ₹{offer_price:,}. "
            "Write a professional opening message that establishes your position."
        )
        
        return self.llm.complete(prompt)
    
    def _generate_acceptance_message(self, accepted_price: int) -> str:
        """Generate acceptance message using LLM"""
        prompt = (
            self.personality_component.make_pre_act_value() +
            f"Context: You are accepting the seller's offer of ₹{accepted_price:,}. "
            "Write a professional acceptance message."
        )
        
        return self.llm.complete(prompt)
    
    def _generate_counter_message(
        self, context: NegotiationContext, seller_price: int, counter_price: int, seller_message: str
    ) -> str:
        """Generate counter-offer message using LLM and memory"""
        
        # Get conversation context from memory
        recent_context = self.memory.get_recent_turns(3)
        negotiation_summary = self.memory.get_negotiation_summary()
        
        rounds_left = 10 - context.current_round
        urgency = "high" if rounds_left <= CFG.late_round else "normal"
        
        prompt = (
            self.personality_component.make_pre_act_value() +
            f"Negotiation context:\n{recent_context}\n"
            f"Summary: {negotiation_summary}\n\n"
            f"Current situation:\n"
            f"- Seller's offer: ₹{seller_price:,}\n"
            f"- Your counter-offer: ₹{counter_price:,}\n" 
            f"- Round {context.current_round}/10 (urgency: {urgency})\n"
            f"- Seller said: '{seller_message}'\n\n"
            "Write a persuasive counter-offer message that justifies your price "
            "without exceeding your numeric offer."
        )
        
        return self.llm.complete(prompt)


# ============================================
# PART 4: EXAMPLE SIMPLE AGENT (FOR REFERENCE)  
# ============================================

class ExampleSimpleAgent(BaseBuyerAgent):
    """Simple example agent for comparison"""
    
    def define_personality(self) -> Dict[str, Any]:
        return {
            "personality_type": "cautious",
            "traits": ["careful", "budget-conscious", "polite"],
            "negotiation_style": "Makes small incremental offers, very careful with money",
            "catchphrases": ["Let me think about that...", "That's a bit steep for me"]
        }
    
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        # Start at 60% of market price
        opening = int(context.product.base_market_price * 0.6)
        opening = min(opening, context.your_budget)
        
        return opening, f"I'm interested, but ₹{opening} is what I can offer. Let me think about that..."
    
    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        # Accept if within budget and below 85% of market
        if seller_price <= context.your_budget and seller_price <= context.product.base_market_price * 0.85:
            return DealStatus.ACCEPTED, seller_price, f"Alright, ₹{seller_price} works for me!"
        
        # Counter with small increment
        last_offer = context.your_offers[-1] if context.your_offers else 0
        counter = min(int(last_offer * 1.1), context.your_budget)
        
        if counter >= seller_price * 0.95:  # Close to agreement
            counter = min(seller_price - 1000, context.your_budget)
            return DealStatus.ONGOING, counter, f"That's a bit steep for me. How about ₹{counter}?"
        
        return DealStatus.ONGOING, counter, f"I can go up to ₹{counter}, but that's pushing my budget."
    
    def get_personality_prompt(self) -> str:
        return """
        I am a cautious buyer who is very careful with money. I speak politely but firmly.
        I often say things like 'Let me think about that' or 'That's a bit steep for me'.
        I make small incremental offers and show concern about my budget.
        """


# ============================================
# PART 5: TESTING FRAMEWORK (DO NOT MODIFY)
# ============================================

class MockSellerAgent:
    """A simple mock seller for testing your agent"""
    
    def __init__(self, min_price: int, personality: str = "standard"):
        self.min_price = min_price
        self.personality = personality
        
    def get_opening_price(self, product: Product) -> Tuple[int, str]:
        # Start at 150% of market price
        price = int(product.base_market_price * 1.5)
        return price, f"These are premium {product.quality_grade} grade {product.name}. I'm asking ₹{price}."
    
    def respond_to_buyer(self, buyer_offer: int, round_num: int) -> Tuple[int, str, bool]:
        if buyer_offer >= self.min_price * 1.1:  # Good profit
            return buyer_offer, f"You have a deal at ₹{buyer_offer}!", True
            
        if round_num >= 8:  # Close to timeout
            counter = max(self.min_price, int(buyer_offer * 1.05))
            return counter, f"Final offer: ₹{counter}. Take it or leave it.", False
        else:
            counter = max(self.min_price, int(buyer_offer * 1.15))
            return counter, f"I can come down to ₹{counter}.", False


def run_negotiation_test(buyer_agent: BaseBuyerAgent, product: Product, buyer_budget: int, seller_min: int) -> Dict[str, Any]:
    """Test a negotiation between your buyer and a mock seller"""
    
    seller = MockSellerAgent(seller_min)
    context = NegotiationContext(
        product=product,
        your_budget=buyer_budget,
        current_round=0,
        seller_offers=[],
        your_offers=[],
        messages=[]
    )
    
    # Seller opens
    seller_price, seller_msg = seller.get_opening_price(product)
    context.seller_offers.append(seller_price)
    context.messages.append({"role": "seller", "message": seller_msg})
    
    # Run negotiation
    deal_made = False
    final_price = None
    
    for round_num in range(10):  # Max 10 rounds
        context.current_round = round_num + 1
        
        # Buyer responds
        if round_num == 0:
            buyer_offer, buyer_msg = buyer_agent.generate_opening_offer(context)
            status = DealStatus.ONGOING
        else:
            status, buyer_offer, buyer_msg = buyer_agent.respond_to_seller_offer(
                context, seller_price, seller_msg
            )
        
        context.your_offers.append(buyer_offer)
        context.messages.append({"role": "buyer", "message": buyer_msg})
        
        if status == DealStatus.ACCEPTED:
            deal_made = True
            final_price = seller_price
            break
            
        # Seller responds
        seller_price, seller_msg, seller_accepts = seller.respond_to_buyer(buyer_offer, round_num)
        
        if seller_accepts:
            deal_made = True
            final_price = buyer_offer
            context.messages.append({"role": "seller", "message": seller_msg})
            break
            
        context.seller_offers.append(seller_price)
        context.messages.append({"role": "seller", "message": seller_msg})
    
    # Calculate results
    result = {
        "deal_made": deal_made,
        "final_price": final_price,
        "rounds": context.current_round,
        "savings": buyer_budget - final_price if deal_made else 0,
        "savings_pct": ((buyer_budget - final_price) / buyer_budget * 100) if deal_made else 0,
        "below_market_pct": ((product.base_market_price - final_price) / product.base_market_price * 100) if deal_made else 0,
        "conversation": context.messages
    }
    
    return result


# ============================================
# PART 6: TEST YOUR AGENT
# ============================================

def test_your_agent():
    """Run this to test your agent implementation"""
    
    # Create test products
    test_products = [
        Product(
            name="Alphonso Mangoes",
            category="Mangoes", 
            quantity=100,
            quality_grade="A",
            origin="Ratnagiri",
            base_market_price=180000,
            attributes={"ripeness": "optimal", "export_grade": True}
        ),
        Product(
            name="Kesar Mangoes",
            category="Mangoes",
            quantity=150,
            quality_grade="B", 
            origin="Gujarat",
            base_market_price=150000,
            attributes={"ripeness": "semi-ripe", "export_grade": False}
        )
    ]
    
    # Initialize your agent
    your_agent = YourBuyerAgent("ConcordiaBuyer")
    
    print("="*60)
    print(f"TESTING CONCORDIA-ENHANCED AGENT: {your_agent.name}")
    print(f"Personality: {your_agent.personality['personality_type']}")
    print(f"Concordia Integration: {'✅ Active' if HAVE_CONCORDIA else '⚠️  Stub Mode'}")
    print("="*60)
    
    total_savings = 0
    deals_made = 0
    
    # Run multiple test scenarios
    for product in test_products:
        for scenario in ["easy", "medium", "hard"]:
            if scenario == "easy":
                buyer_budget = int(product.base_market_price * 1.2)
                seller_min = int(product.base_market_price * 0.8)
            elif scenario == "medium":
                buyer_budget = int(product.base_market_price * 1.0)
                seller_min = int(product.base_market_price * 0.85)
            else:  # hard
                buyer_budget = int(product.base_market_price * 0.9)
                seller_min = int(product.base_market_price * 0.82)
            
            print(f"\nTest: {product.name} - {scenario.upper()} scenario")
            print(f"Budget: ₹{buyer_budget:,} | Market: ₹{product.base_market_price:,} | Seller Min: ₹{seller_min:,}")
            
            result = run_negotiation_test(your_agent, product, buyer_budget, seller_min)
            
            if result["deal_made"]:
                deals_made += 1
                total_savings += result["savings"]
                print(f"✅ SUCCESS: Deal at ₹{result['final_price']:,} in {result['rounds']} rounds")
                print(f"   💰 Savings: ₹{result['savings']:,} ({result['savings_pct']:.1f}% under budget)")
                print(f"   📊 Below Market: {result['below_market_pct']:.1f}%")
            else:
                print(f"❌ FAILED: No deal after {result['rounds']} rounds")
    
    # Summary
    print("\n" + "="*60)
    print("🎯 FINAL RESULTS")
    print(f"✅ Deals Completed: {deals_made}/6 ({deals_made/6*100:.1f}% success rate)")
    print(f"💰 Total Savings: ₹{total_savings:,}")
    print(f"📈 Average Savings per Deal: ₹{total_savings//max(deals_made,1):,}")
    print("="*60)
    
    return deals_made, total_savings


# ============================================
# PART 7: CONCORDIA FACTORY FUNCTION
# ============================================

def make_buyer_agent(cfg: Optional[Dict[str, Any]] = None) -> YourBuyerAgent:
    """
    Factory function for Concordia compatibility.
    Creates a buyer agent that works with both Concordia and template systems.
    """
    cfg = cfg or {}
    return YourBuyerAgent(
        name=cfg.get("name", "ConcordiaBuyer"),
        model_name=cfg.get("model", "llama3:8b"),
    )


if __name__ == "__main__":
    print("🚀 CONCORDIA-ENHANCED NEGOTIATION AGENT")
    print("=" * 50)
    
    # Run comprehensive test
    deals, savings = test_your_agent()
    
    # Compare with simple agent
    print("\n" + "🔄 COMPARISON WITH SIMPLE AGENT")
    print("=" * 50)
    
    simple_agent = ExampleSimpleAgent("SimpleAgent")
    simple_deals = 0
    simple_savings = 0
    
    test_products = [
        Product("Alphonso Mangoes", "Mangoes", 100, "A", "Ratnagiri", 180000, {}),
        Product("Kesar Mangoes", "Mangoes", 150, "B", "Gujarat", 150000, {})
    ]
    
    for product in test_products:
        for scenario in ["easy", "medium", "hard"]:
            if scenario == "easy":
                buyer_budget = int(product.base_market_price * 1.2)
                seller_min = int(product.base_market_price * 0.8)
            elif scenario == "medium":
                buyer_budget = int(product.base_market_price * 1.0)
                seller_min = int(product.base_market_price * 0.85)
            else:
                buyer_budget = int(product.base_market_price * 0.9)
                seller_min = int(product.base_market_price * 0.82)
            
            result = run_negotiation_test(simple_agent, product, buyer_budget, seller_min)
            if result["deal_made"]:
                simple_deals += 1
                simple_savings += result["savings"]
    
    print(f"Simple Agent: {simple_deals}/6 deals, ₹{simple_savings:,} total savings")
    print(f"Concordia Agent: {deals}/6 deals, ₹{savings:,} total savings")
    print(f"Improvement: +{deals - simple_deals} deals, +₹{savings - simple_savings:,} savings")