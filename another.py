"""
===========================================
AI NEGOTIATION AGENT - CONCORDIA ENHANCED
===========================================

Buyer agent that integrates Concordia framework with intelligent seller opponent.
Features personality-driven negotiations, memory-aware decisions, and full conversation display.
"""

import json
import re
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import random
import argparse
import sys
from datetime import datetime

# ============================================================
# Graceful Concordia imports ─ fall back to local stubs
# ============================================================
try:
    from concordia.typing import entity_component
 

    # Associative memory
    from concordia.associative_memory import associative_memory

    # Language model
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
            def __init__(self, *args, **kwargs): 
                self._buf: List[Dict[str, Any]] = []
            def add_observation(self, obj): 
                self._buf.append(obj)
            def retrieve(self, k: int = 5): 
                return self._buf[-k:]

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

from sentence_transformers import SentenceTransformer

class SBertEmbedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

class _ConcordiaMemory(associative_memory.AssociativeMemory):  # type: ignore[misc]
    def __init__(self, *args, **kwargs):
        super().__init__(sentence_embedder=SBertEmbedder(), *args, **kwargs)
        self._conversation_buffer: List[Dict[str, Any]] = []


    def add_conversation_turn(self, role: str, message: str, price: Optional[int] = None):
        """Add a conversation turn to memory"""
        entry = {"role": role, "text": message}
        if price is not None:
            entry["price"] = price
        self._conversation_buffer.append(entry)
        # Use add_observation if available, otherwise just store locally
        try:
            self.add_observation(entry)
        except AttributeError:
            pass  # Fallback mode

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

class _PersonalityComponent(entity_component.ContextComponent):  # type: ignore[misc]
    """Concordia personality component"""
    
    def __init__(self, agent_type: str = "buyer"):
        super().__init__()
        if agent_type == "buyer":
            self.type = "analytical-diplomatic"
            self.traits = ["calm", "strategic", "data-driven", "fair"]
            self.catchphrases = [
                "Let's be fair to both sides.",
                "I've done my research.", 
                "We can find a middle ground."
            ]
            self.style = "professional and respectful"
        else:  # seller
            self.type = "experienced-persuasive"
            self.traits = ["confident", "persuasive", "quality-focused", "profit-aware"]
            self.catchphrases = [
                "Quality comes at a price.",
                "This is premium merchandise.",
                "I know the market well."
            ]
            self.style = "confident but flexible"
        
        self.agent_type = agent_type

    def make_pre_act_value(self) -> str:
        """Generate personality context for LLM prompts"""
        traits_str = ", ".join(self.traits)
        phrases_str = ", ".join(self.catchphrases)
        return (
            f"You are a {self.type} {self.agent_type} agent. Your traits: {traits_str}. "
            f"Your communication style is {self.style}. "
            f"Use phrases like: {phrases_str}. "
            "Speak in 1-2 professional, concise sentences. "
            f"Never exceed your numeric {'budget' if self.agent_type == 'buyer' else 'minimum price'} constraints.\n"
        )

    # 🔧 Required abstract methods
    def get_state(self) -> dict:
        """Return current state of personality"""
        return {
            "agent_type": self.agent_type,
            "type": self.type,
            "traits": self.traits,
            "catchphrases": self.catchphrases,
            "style": self.style
        }

    def set_state(self, state: dict) -> None:
        """Restore personality state"""
        self.agent_type = state.get("agent_type", self.agent_type)
        self.type = state.get("type", self.type)
        self.traits = state.get("traits", self.traits)
        self.catchphrases = state.get("catchphrases", self.catchphrases)
        self.style = state.get("style", self.style)


class _OllamaLLM(language_model.LanguageModel):  # type: ignore[misc]
    """Ollama LLM wrapper with UTF-8 safety"""

    def __init__(self, model: str = "llama3:8b", timeout: int = 60):
        super().__init__()
        self.model = model
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        """Generate response using Ollama"""
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
            return response if response else f"I understand. Let me respond appropriately."
        except Exception as e:
            # Fallback for environments without Ollama
            return f"Based on the context, I'll proceed with the negotiation."

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
        self.personality_component = _PersonalityComponent("buyer")
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
# PART 4: CONCORDIA-ENHANCED SELLER AGENT
# ============================================

class ConcordiaSellerAgent:
    """
    Intelligent Concordia-based seller agent with personality and memory
    """
    
    def __init__(self, min_price: int, model_name: str = "llama3:8b"):
        self.name = "ConcordiaSeller"
        self.min_price = min_price
        
        # Concordia components
        self.personality_component = _PersonalityComponent("seller")
        self.memory = _ConcordiaMemory()
        self.llm = _OllamaLLM(model_name)
        
        # Seller-specific strategy
        self.initial_markup = 1.5  # Start at 150% of market
        self.min_margin = 0.1  # Minimum 10% profit margin
        self.concession_rate = 0.15  # How much to concede each round
        
    def get_opening_price(self, product: Product) -> Tuple[int, str]:
        """Generate opening offer with personality"""
        opening_price = max(int(product.base_market_price * self.initial_markup), self.min_price)
        
        message = self._generate_opening_message(product, opening_price)
        self.memory.add_conversation_turn("seller", message, opening_price)
        
        return opening_price, message
        
    def respond_to_buyer(self, buyer_offer: int, buyer_message: str, round_num: int, product: Product) -> Tuple[int, str, bool]:
        """Intelligent response to buyer offers"""
        
        # Store buyer's offer in memory
        self.memory.add_conversation_turn("buyer", buyer_message, buyer_offer)
        
        # Decision logic
        profit_margin = (buyer_offer - self.min_price) / self.min_price
        rounds_left = 10 - round_num
        
        # Accept if good profit or near deadline with acceptable offer
        if (profit_margin >= self.min_margin and buyer_offer >= self.min_price * 1.05) or \
           (rounds_left <= 2 and buyer_offer >= self.min_price):
            message = self._generate_acceptance_message(buyer_offer)
            self.memory.add_conversation_turn("seller", message, buyer_offer)
            return buyer_offer, message, True
        
        # Calculate counter offer
        if rounds_left <= 3:  # Urgent concessions
            counter_price = max(self.min_price, int(buyer_offer * 1.02))
        elif rounds_left <= 6:  # Moderate concessions
            counter_price = max(self.min_price, int(buyer_offer * 1.08))
        else:  # Conservative concessions
            counter_price = max(self.min_price, int(buyer_offer * 1.12))
        
        # Generate response message
        message = self._generate_counter_message(buyer_offer, counter_price, buyer_message, round_num, product)
        self.memory.add_conversation_turn("seller", message, counter_price)
        
        return counter_price, message, False
    
    def _generate_opening_message(self, product: Product, opening_price: int) -> str:
        """Generate opening message using LLM"""
        prompt = (
            self.personality_component.make_pre_act_value() +
            f"Context: You're selling {product.quantity}kg of {product.quality_grade}-grade "
            f"{product.name} from {product.origin}. Market price is ₹{product.base_market_price:,}. "
            f"Your opening price: ₹{opening_price:,}. "
            "Write a compelling opening sales message highlighting quality and value."
        )
        
        return self.llm.complete(prompt)
    
    def _generate_acceptance_message(self, accepted_price: int) -> str:
        """Generate acceptance message using LLM"""
        prompt = (
            self.personality_component.make_pre_act_value() +
            f"Context: You are accepting the buyer's offer of ₹{accepted_price:,}. "
            "Write a professional acceptance message showing satisfaction with the deal."
        )
        
        return self.llm.complete(prompt)
    
    def _generate_counter_message(self, buyer_offer: int, counter_price: int, buyer_message: str, round_num: int, product: Product) -> str:
        """Generate counter-offer message using LLM and memory"""
        
        recent_context = self.memory.get_recent_turns(3)
        rounds_left = 10 - round_num
        urgency = "high" if rounds_left <= 3 else "normal"
        
        prompt = (
            self.personality_component.make_pre_act_value() +
            f"Negotiation context:\n{recent_context}\n\n"
            f"Current situation:\n"
            f"- Buyer's offer: ₹{buyer_offer:,}\n"
            f"- Your counter-offer: ₹{counter_price:,}\n"
            f"- Round {round_num}/10 (urgency: {urgency})\n"
            f"- Product: {product.quality_grade}-grade {product.name}\n"
            f"- Buyer said: '{buyer_message}'\n\n"
            "Write a persuasive counter-offer message that justifies your price "
            "while showing some flexibility. Emphasize product quality and market value."
        )
        
        return self.llm.complete(prompt)

# ============================================
# CONVERSATION DISPLAY UTILITIES
# ============================================

def print_conversation_header(round_num: int, total_rounds: int):
    """Print conversation round header"""
    print(f"\n{'='*80}")
    print(f"🗣️  ROUND {round_num}/{total_rounds}")
    print(f"{'='*80}")

def print_message(role: str, price: int, message: str, status: str = ""):
    """Print formatted conversation message"""
    role_icon = "🏪" if role.upper() == "SELLER" else "🛒"
    status_icon = ""
    if status == "ACCEPTED":
        status_icon = " ✅"
    elif status == "FINAL":
        status_icon = " ⚡"
    
    print(f"\n{role_icon} {role.upper()}{status_icon}")
    print(f"💰 Price: ₹{price:,}")
    print(f"💬 Message: {message}")
    print(f"{'-'*80}")

def print_negotiation_summary(result: Dict[str, Any], product: Product, buyer_budget: int):
    """Print detailed negotiation summary"""
    print(f"\n{'🎯 NEGOTIATION SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"📦 Product: {product.name} ({product.quality_grade}-grade, {product.quantity}kg)")
    print(f"📍 Origin: {product.origin}")
    print(f"📊 Market Price: ₹{product.base_market_price:,}")
    print(f"💰 Buyer Budget: ₹{buyer_budget:,}")
    print(f"🔄 Rounds: {result['rounds']}")
    
    if result['deal_made']:
        print(f"✅ DEAL SUCCESSFUL at ₹{result['final_price']:,}")
        print(f"💰 Buyer Savings: ₹{result['savings']:,} ({result['savings_pct']:.1f}% under budget)")
        print(f"📉 Below Market: {result['below_market_pct']:.1f}%")
    else:
        print(f"❌ NEGOTIATION FAILED - No deal reached")
    
    print(f"{'='*80}")

# ============================================
# PART 5: ENHANCED TESTING FRAMEWORK
# ============================================

def run_negotiation_test(buyer_agent: BaseBuyerAgent, product: Product, buyer_budget: int, seller_min: int, show_conversation: bool = True) -> Dict[str, Any]:
    """Test a negotiation with full conversation display"""
    
    # Create intelligent Concordia seller
    seller = ConcordiaSellerAgent(seller_min)
    context = NegotiationContext(
        product=product,
        your_budget=buyer_budget,
        current_round=0,
        seller_offers=[],
        your_offers=[],
        messages=[]
    )
    
    if show_conversation:
        print(f"\n{'🚀 NEGOTIATION START':^80}")
        print(f"{'='*80}")
        print(f"📦 Product: {product.name} ({product.quality_grade}-grade)")
        print(f"📊 Market Price: ₹{product.base_market_price:,}")
        print(f"💰 Buyer Budget: ₹{buyer_budget:,}")
        print(f"🏪 Seller Minimum: ₹{seller_min:,}")
    
    # Seller opens
    seller_price, seller_msg = seller.get_opening_price(product)
    context.seller_offers.append(seller_price)
    context.messages.append({"role": "seller", "message": seller_msg})
    
    if show_conversation:
        print_conversation_header(1, 10)
        print_message("SELLER", seller_price, seller_msg)
    
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
        
        if show_conversation:
            status_text = "ACCEPTED" if status == DealStatus.ACCEPTED else ""
            print_message("BUYER", buyer_offer, buyer_msg, status_text)
        
        if status == DealStatus.ACCEPTED:
            deal_made = True
            final_price = seller_price
            break
            
        # Seller responds
        seller_price, seller_msg, seller_accepts = seller.respond_to_buyer(
            buyer_offer, buyer_msg, round_num, product
        )
        
        if seller_accepts:
            deal_made = True
            final_price = buyer_offer
            context.messages.append({"role": "seller", "message": seller_msg})
            if show_conversation:
                print_message("SELLER", final_price, seller_msg, "ACCEPTED")
            break
            
        context.seller_offers.append(seller_price)
        context.messages.append({"role": "seller", "message": seller_msg})
        
        if show_conversation and round_num < 9:  # Don't print if it's the last round
            print_conversation_header(round_num + 2, 10)
            status_text = "FINAL" if round_num >= 7 else ""
            print_message("SELLER", seller_price, seller_msg, status_text)
    
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
    
    if show_conversation:
        print_negotiation_summary(result, product, buyer_budget)
    
    return result

def test_your_agent(show_conversations: bool = True):
    """Run comprehensive agent testing with conversation display"""
    
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
    print(f"🧠 TESTING CONCORDIA-ENHANCED AGENTS")
    print(f"👤 Buyer: {your_agent.name} ({your_agent.personality['personality_type']})")
    print(f"🏪 Seller: ConcordiaSeller (experienced-persuasive)")
    print(f"🔧 Concordia: {'✅ Active' if HAVE_CONCORDIA else '⚠️  Stub Mode'}")
    print("="*60)
    
    total_savings = 0
    deals_made = 0
    scenario_count = 0
    
    # Run multiple test scenarios
    for product in test_products:
        for scenario in ["easy", "medium", "hard"]:
            scenario_count += 1
            
            if scenario == "easy":
                buyer_budget = int(product.base_market_price * 1.2)
                seller_min = int(product.base_market_price * 0.8)
            elif scenario == "medium":
                buyer_budget = int(product.base_market_price * 1.0)
                seller_min = int(product.base_market_price * 0.85)
            else:  # hard
                buyer_budget = int(product.base_market_price * 0.9)
                seller_min = int(product.base_market_price * 0.82)
            
            print(f"\n{'🎯 TEST SCENARIO ' + str(scenario_count):^80}")
            print(f"📦 {product.name} - {scenario.upper()} difficulty")
            
            result = run_negotiation_test(
                your_agent, product, buyer_budget, seller_min, 
                show_conversation=show_conversations
            )
            
            if result["deal_made"]:
                deals_made += 1
                total_savings += result["savings"]
    
    # Final summary
    print(f"\n{'🏆 FINAL RESULTS':^80}")
    print(f"{'='*80}")
    print(f"✅ Success Rate: {deals_made}/{scenario_count} ({deals_made/scenario_count*100:.1f}%)")
    print(f"💰 Total Savings: ₹{total_savings:,}")

# ============================================
# PART 6: COMMAND LINE INTERFACE & UTILITIES
# ============================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Concordia-Enhanced AI Negotiation Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python negotiation_agent.py                    # Run all tests with conversations
  python negotiation_agent.py --silent           # Run tests without conversation display
  python negotiation_agent.py --single           # Run single negotiation
  python negotiation_agent.py --model llama2     # Use different LLM model
        """
    )
    
    parser.add_argument(
        "--silent", 
        action="store_true",
        help="Run tests without showing conversation details"
    )
    
    parser.add_argument(
        "--single",
        action="store_true", 
        help="Run a single interactive negotiation"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="llama3:8b",
        help="LLM model to use (default: llama3:8b)"
    )
    
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Maximum negotiation rounds (default: 10)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with extra information"
    )
    
    return parser.parse_args()

def run_single_negotiation(model_name: str = "llama3:8b", max_rounds: int = 10):
    """Run a single interactive negotiation"""
    
    print("🎯 SINGLE NEGOTIATION MODE")
    print("=" * 50)
    
    # Create sample product
    product = Product(
        name="Premium Basmati Rice",
        category="Grains",
        quantity=500,
        quality_grade="Export",
        origin="Punjab",
        base_market_price=200000,
        attributes={"aging": "2_years", "moisture": "12%"}
    )
    
    print(f"📦 Product: {product.name}")
    print(f"📊 Market Price: ₹{product.base_market_price:,}")
    print(f"⚙️  Model: {model_name}")
    print(f"🔄 Max Rounds: {max_rounds}")
    
    # Get user inputs
    try:
        buyer_budget = int(input(f"\n💰 Enter buyer budget (market: ₹{product.base_market_price:,}): ₹"))
        seller_min = int(input(f"🏪 Enter seller minimum price: ₹"))
        
        if buyer_budget <= 0 or seller_min <= 0:
            print("❌ Invalid prices. Please enter positive numbers.")
            return
            
        if seller_min > buyer_budget:
            print("⚠️  Warning: Seller minimum is higher than buyer budget. Deal unlikely.")
            
    except ValueError:
        print("❌ Invalid input. Please enter numeric values.")
        return
    
    # Create agents
    buyer_agent = YourBuyerAgent("InteractiveBuyer", model_name)
    
    # Run negotiation
    result = run_negotiation_test(buyer_agent, product, buyer_budget, seller_min, show_conversation=True)
    
    # Show detailed results
    print("\n🎯 DETAILED ANALYSIS")
    print("=" * 50)
    
    if result["deal_made"]:
        efficiency = (result["savings"] / buyer_budget) * 100
        print(f"✅ Negotiation Efficiency: {efficiency:.1f}%")
        print(f"📈 Price Movement: {len(result['conversation'])//2} rounds")
        print(f"🎪 Final Deal: ₹{result['final_price']:,}")
    else:
        print(f"❌ Gap too large: Budget ₹{buyer_budget:,} vs Min ₹{seller_min:,}")

def validate_environment():
    """Validate the environment and dependencies"""
    issues = []
    
    # Check Concordia availability
    if not HAVE_CONCORDIA:
        issues.append("⚠️  Concordia framework not installed - using stub mode")
    
    # Check Ollama availability
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, timeout=5)
    except Exception:
        issues.append("⚠️  Ollama not available - using fallback responses")
    
    if issues:
        print("🔧 ENVIRONMENT STATUS")
        print("=" * 30)
        for issue in issues:
            print(issue)
        print("💡 The agent will still work with reduced functionality.\n")
    
    return len(issues) == 0

def run_performance_benchmark(model_name: str = "llama3:8b"):
    """Run performance benchmark across multiple scenarios"""
    
    print("🏃‍♂️ PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    scenarios = [
        {"name": "Easy Wins", "budget_mult": 1.3, "seller_mult": 0.75},
        {"name": "Fair Deals", "budget_mult": 1.1, "seller_mult": 0.85}, 
        {"name": "Tough Negotiations", "budget_mult": 0.95, "seller_mult": 0.88},
        {"name": "Extreme Pressure", "budget_mult": 0.85, "seller_mult": 0.90}
    ]
    
    products = [
        Product("Alphonso Mangoes", "Mangoes", 100, "A", "Ratnagiri", 180000),
        Product("Kesar Mangoes", "Mangoes", 150, "B", "Gujarat", 150000),
        Product("Basmati Rice", "Grains", 500, "Export", "Punjab", 200000),
        Product("Darjeeling Tea", "Tea", 50, "A", "West Bengal", 80000)
    ]
    
    agent = YourBuyerAgent("BenchmarkAgent", model_name)
    total_tests = len(scenarios) * len(products)
    
    results = {
        "total_tests": total_tests,
        "successful_deals": 0,
        "total_savings": 0,
        "avg_rounds": 0,
        "scenario_breakdown": {}
    }
    
    test_count = 0
    total_rounds = 0
    
    for scenario in scenarios:
        scenario_results = {"deals": 0, "tests": 0, "savings": 0}
        
        for product in products:
            test_count += 1
            buyer_budget = int(product.base_market_price * scenario["budget_mult"])
            seller_min = int(product.base_market_price * scenario["seller_mult"])
            
            print(f"🔄 Test {test_count}/{total_tests}: {scenario['name']} - {product.name}")
            
            result = run_negotiation_test(agent, product, buyer_budget, seller_min, show_conversation=False)
            
            scenario_results["tests"] += 1
            total_rounds += result["rounds"]
            
            if result["deal_made"]:
                results["successful_deals"] += 1
                results["total_savings"] += result["savings"]
                scenario_results["deals"] += 1
                scenario_results["savings"] += result["savings"]
                print(f"   ✅ Deal at ₹{result['final_price']:,} (saved ₹{result['savings']:,})")
            else:
                print(f"   ❌ No deal")
        
        results["scenario_breakdown"][scenario["name"]] = scenario_results
    
    results["avg_rounds"] = total_rounds / total_tests
    
    # Print benchmark results
    print(f"\n{'🏆 BENCHMARK RESULTS':^60}")
    print("=" * 60)
    print(f"Success Rate: {results['successful_deals']}/{total_tests} ({results['successful_deals']/total_tests*100:.1f}%)")
    print(f"Total Savings: ₹{results['total_savings']:,}")
    print(f"Average Rounds: {results['avg_rounds']:.1f}")
    
    print(f"\n{'Scenario Breakdown':^40}")
    print("-" * 40)
    for scenario_name, data in results["scenario_breakdown"].items():
        success_rate = (data["deals"] / data["tests"]) * 100
        print(f"{scenario_name:20} {data['deals']}/{data['tests']} ({success_rate:5.1f}%) ₹{data['savings']:,}")
    
    return results

def export_negotiation_log(result: Dict[str, Any], filename: Optional[str] = None):
    """Export negotiation conversation to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"negotiation_log_{timestamp}.json"
    
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "result": result,
        "concordia_available": HAVE_CONCORDIA
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Negotiation log exported to: {filename}")

def load_custom_product(product_file: str) -> Product:
    """Load product from JSON file"""
    with open(product_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return Product(**data)

def create_personality_report(agent: YourBuyerAgent) -> str:
    """Generate detailed personality analysis report"""
    personality = agent.personality
    
    report = f"""
🧠 AGENT PERSONALITY REPORT
{'='*50}

Agent Name: {agent.name}
Personality Type: {personality.get('personality_type', 'Unknown')}

Core Traits:
{chr(10).join(f"  • {trait}" for trait in personality.get('traits', []))}

Negotiation Style:
{personality.get('negotiation_style', 'No style defined')}

Communication Patterns:
{chr(10).join(f"  • {phrase}" for phrase in personality.get('catchphrases', []))}

Concordia Integration:
  • Memory Component: {'✅ Active' if hasattr(agent, 'memory') else '❌ Missing'}
  • Personality Component: {'✅ Active' if hasattr(agent, 'personality_component') else '❌ Missing'}
  • LLM Integration: {'✅ Active' if hasattr(agent, 'llm') else '❌ Missing'}
  • Policy Engine: {'✅ Active' if hasattr(agent, 'policy') else '❌ Missing'}
"""
    
    return report

def main():
    """Main execution function"""
    args = parse_arguments()
    
    print("🤖 CONCORDIA-ENHANCED AI NEGOTIATION AGENT")
    print("=" * 60)
    print(f"🔧 Concordia Status: {'✅ Active' if HAVE_CONCORDIA else '⚠️  Stub Mode'}")
    print(f"🧠 LLM Model: {args.model}")
    
    # Validate environment
    validate_environment()
    
    try:
        if args.single:
            run_single_negotiation(args.model, args.rounds)
        elif args.debug:
            run_performance_benchmark(args.model)
        else:
            test_your_agent(show_conversations=not args.silent)
            
    except KeyboardInterrupt:
        print("\n🛑 Negotiation interrupted by user")
    except Exception as e:
        if args.debug:
            raise
        else:
            print(f"❌ Error: {e}")
            print("💡 Use --debug flag for detailed error information")

# ============================================
# FINAL EXECUTION
# ============================================

if __name__ == "__main__":
    main()
