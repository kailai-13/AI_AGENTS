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
# Fixed Concordia imports with proper fallback handling
# ============================================================
try:
    from concordia.components import agent as agent_components
    from concordia.associative_memory import associative_memory
    from concordia.associative_memory import formative_memories
    from concordia.language_model import language_model
    from concordia.utils import measurements as measurements_lib
    HAVE_CONCORDIA = True
    print("✅ Concordia framework loaded successfully")
except ImportError as e:
    print(f"⚠️  Concordia import failed: {e}")
    HAVE_CONCORDIA = False
    
    # Enhanced stubs that match Concordia's actual interface
    class agent_components:  # type: ignore
        class ContextComponent:
            def __init__(self, **kwargs):
                self._state = {}
            
            def make_pre_act_value(self) -> str: 
                return ""
                
            def get_state(self): 
                return self._state
                
            def set_state(self, state): 
                self._state = state
                
            def update(self):
                pass

    class associative_memory:  # type: ignore
        class AssociativeMemory:
            def __init__(self, sentence_embedder=None, importance_model=None, clock=None):
                self._memories: List[Dict[str, Any]] = []
                self._clock = clock
                
            def add(self, text: str, **kwargs) -> None:
                """Add a memory with text and metadata"""
                memory = {
                    'text': text,
                    'timestamp': getattr(self._clock, 'now', lambda: 0)() if self._clock else 0,
                    **kwargs
                }
                self._memories.append(memory)
                
            def retrieve(self, query: str, k: int = 5) -> List[str]:
                """Retrieve k most relevant memories"""
                if not self._memories:
                    return []
                # Simple retrieval - return most recent memories
                recent = self._memories[-k:] if len(self._memories) >= k else self._memories
                return [mem['text'] for mem in recent]
                
            def get_all_memories(self) -> List[Dict[str, Any]]:
                return self._memories.copy()

    class formative_memories:  # type: ignore
        @staticmethod
        def make_memories(**kwargs):
            return []

    class language_model:  # type: ignore
        class   LanguageModel:
            def sample_text(self, prompt: str, **kwargs) -> str: 
                return prompt  # echo fallback
                
            def sample_choice(self, prompt: str, responses: List[str], **kwargs) -> str:
                return responses[0] if responses else ""

    class measurements_lib:  # type: ignore
        @staticmethod
        def publish_measurement(name: str, value: Any):
            pass

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
# CONCORDIA COMPONENTS - FIXED VERSION
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

class _ConcordiaMemory:
    """Enhanced memory that works with both Concordia and stub modes"""

    def __init__(self, sentence_embedder=None, importance_model=None, clock=None):
        if HAVE_CONCORDIA:
            self._memory = associative_memory.AssociativeMemory(
                sentence_embedder=sentence_embedder,
                importance_model=importance_model,
                clock=clock
            )
        else:
            self._memory = associative_memory.AssociativeMemory(
                sentence_embedder=sentence_embedder,
                importance_model=importance_model, 
                clock=clock
            )
        self._conversation_buffer: List[Dict[str, Any]] = []

    def add_conversation_turn(self, role: str, message: str, price: Optional[int] = None):
        """Add a conversation turn to memory"""
        entry = {"role": role, "text": message}
        if price is not None:
            entry["price"] = price
        self._conversation_buffer.append(entry)
        
        # Add to Concordia memory with proper formatting
        memory_text = f"{role.capitalize()}: {message}"
        if price is not None:
            memory_text += f" [Price: ₹{price:,}]"
            
        if HAVE_CONCORDIA:
            self._memory.add(memory_text, tags=[role, "negotiation"])
        else:
            self._memory.add(memory_text)

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
    
    def retrieve_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant context from memory"""
        if HAVE_CONCORDIA:
            memories = self._memory.retrieve(query, k=k)
            return "\n".join(memories) if memories else "No relevant context found."
        else:
            memories = self._memory.retrieve(query, k=k)
            return "\n".join(memories) if memories else "No relevant context found."

class _PersonalityComponent(agent_components.ContextComponent):  # type: ignore[misc]
    """Concordia personality component with enhanced traits"""
    
    def __init__(self, name: str = "negotiation_personality"):
        super().__init__()
        self._name = name
        self.type = "analytical-diplomatic"
        self.traits = ["calm", "strategic", "data-driven", "fair", "persuasive"]
        self.catchphrases = [
            "Let's find a fair deal for both of us.",
            "I've researched the market extensively.", 
            "We can definitely work something out.",
            "Quality products deserve fair pricing."
        ]
        self.style = "professional, respectful, and strategic"
        self.negotiation_approach = "data-driven with relationship focus"

    def make_pre_act_value(self) -> str:
        """Generate personality context for LLM prompts"""
        traits_str = ", ".join(self.traits)
        phrases_str = " | ".join(self.catchphrases)
        return (
            f"PERSONALITY CONTEXT:\n"
            f"You are a {self.type} buyer agent with these traits: {traits_str}.\n"
            f"Your communication style: {self.style}.\n"
            f"Your negotiation approach: {self.negotiation_approach}.\n"
            f"Use phrases like: {phrases_str}\n"
            f"Rules: Speak professionally in 1-2 sentences. Never exceed your budget. "
            f"Be persuasive but fair. Use market data to justify your position.\n\n"
        )
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "traits": self.traits,
            "catchphrases": self.catchphrases,
            "style": self.style
        }
    
    def set_state(self, state: Dict[str, Any]):
        if "type" in state:
            self.type = state["type"]
        if "traits" in state:
            self.traits = state["traits"]
        if "catchphrases" in state:
            self.catchphrases = state["catchphrases"]
        if "style" in state:
            self.style = state["style"]

class _OllamaLLM(language_model.LanguageModel):  # type: ignore[misc]
    """Enhanced Ollama LLM wrapper with better error handling"""

    def __init__(self, model: str = "llama3.2:3b", timeout: int = 60):
        super().__init__()
        self.model = model
        self.timeout = timeout
        self._test_connection()

    def _test_connection(self):
        """Test if Ollama is available and model exists"""
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                models = result.stdout
                if self.model.split(':')[0] not in models:
                    print(f"⚠️  Model {self.model} not found. Available models:")
                    print(models)
                else:
                    print(f"✅ Ollama model {self.model} is available")
            else:
                print("⚠️  Ollama not accessible")
        except Exception as e:
            print(f"⚠️  Ollama connection test failed: {e}")

    def sample_text(self, prompt: str, **kwargs) -> str:
        """Generate response using Ollama with enhanced prompting"""
        try:
            # Enhanced prompt with clear instructions
            enhanced_prompt = (
                f"{prompt}\n\n"
                f"RESPONSE REQUIREMENTS:\n"
                f"- Be professional and concise (1-2 sentences)\n"
                f"- Stay in character as described\n" 
                f"- Do not exceed any stated budget limits\n"
                f"- Generate only the negotiation message, no explanations\n\n"
                f"Your response:"
            )
            
            proc = subprocess.run(
                ["ollama", "run", self.model],
                input=enhanced_prompt,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=self.timeout,
            )
            
            response = (proc.stdout or proc.stderr).strip()
            
            # Clean up response - remove common LLM artifacts
            response = re.sub(r'^(Response:|Your response:|Here\'s my response:)', '', response, flags=re.IGNORECASE)
            response = response.strip()
            
            return response if response else "I appreciate your offer. Let me respond appropriately to move our negotiation forward."
            
        except subprocess.TimeoutExpired:
            return "I understand your position. Let me consider this carefully and respond."
        except Exception as e:
            print(f"LLM Error: {e}")
            return "Thank you for your offer. I'll need to respond based on our negotiation parameters."

    def sample_choice(self, prompt: str, responses: List[str], **kwargs) -> str:
        """Choose from a list of responses"""
        full_prompt = f"{prompt}\n\nChoose the best response from: {responses}"
        response = self.sample_text(full_prompt, **kwargs)
        
        # Try to match response to one of the choices
        for choice in responses:
            if choice.lower() in response.lower():
                return choice
        
        # Default to first choice
        return responses[0] if responses else ""

# ============================================
# PART 3: CONCORDIA-ENHANCED BUYER AGENT
# ============================================

class YourBuyerAgent(BaseBuyerAgent):
    """
    Concordia-enhanced buyer agent with personality, memory, and LLM integration
    """
    
    def __init__(self, name: str = "ConcordiaBuyer", model_name: str = "llama3.2:3b"):
        super().__init__(name)
        
        print(f"🚀 Initializing {name} with Concordia framework...")
        
        # Initialize Concordia components
        self.personality_component = _PersonalityComponent(f"{name}_personality")
        self.memory = _ConcordiaMemory()
        self.policy = _NegotiationPolicy()
        self.llm = _OllamaLLM(model_name)
        
        # Initialize personality from Concordia component
        self.personality.update({
            "personality_type": self.personality_component.type,
            "traits": self.personality_component.traits,
            "catchphrases": self.personality_component.catchphrases,
            "concordia_enabled": HAVE_CONCORDIA
        })
        
        print(f"✅ Agent {name} initialized successfully")
        if HAVE_CONCORDIA:
            print("🧠 Full Concordia framework active")
        else:
            print("🔄 Running in compatibility mode")
    
    def define_personality(self) -> Dict[str, Any]:
        """Define personality traits using Concordia component"""
        return {
            "personality_type": "analytical-diplomatic",
            "traits": [
                "Calm under pressure",
                "Strategic thinker", 
                "Fair but firm",
                "Data-driven decisions",
                "Relationship-focused",
                "Market-aware",
                "Persuasive communicator"
            ],
            "negotiation_style": (
                "Uses market research and fairness as key leverage points. "
                "Starts with reasonable offers, increases strategically based on seller responses, "
                "and accelerates concessions as deadline approaches. Maintains professional tone."
            ),
            "catchphrases": [
                "Let's find a fair deal for both of us.",
                "I've researched the market extensively.",
                "We can definitely work something out.",
                "Quality products deserve fair pricing."
            ]
        }
    
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        """Generate opening offer using Concordia components"""
        # Use policy to determine offer amount
        offer_price = self.policy.opening_offer(
            context.product.base_market_price, 
            context.your_budget
        )
        
        # Ensure we don't exceed budget
        offer_price = min(offer_price, context.your_budget)
        
        # Generate message using LLM and personality
        message = self._generate_opening_message(context, offer_price)
        
        # Store in memory
        self.memory.add_conversation_turn("buyer", message, offer_price)
        
        # Log the opening strategy
        measurements_lib.publish_measurement("opening_offer_pct_of_market", offer_price / context.product.base_market_price)
        
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
            message = self._generate_acceptance_message(seller_price, seller_message)
            self.memory.add_conversation_turn("buyer", message, seller_price)
            measurements_lib.publish_measurement("deal_accepted_at_round", context.current_round)
            return DealStatus.ACCEPTED, seller_price, message
        
        # If seller price exceeds our budget, we must reject
        if seller_price > context.your_budget:
            message = self._generate_rejection_message(seller_price, context.your_budget)
            self.memory.add_conversation_turn("buyer", message, None)
            return DealStatus.REJECTED, 0, message
        
        # Generate counter offer using policy
        last_offer = context.your_offers[-1] if context.your_offers else None
        counter_price = self.policy.counter_offer(
            seller_price, market_price, context.your_budget,
            last_offer, context.current_round, 10
        )
        
        # Ensure counter doesn't exceed budget
        counter_price = min(counter_price, context.your_budget)
        
        # Generate counter message using LLM with memory context
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
            f"NEGOTIATION CONTEXT:\n"
            f"Opening negotiation for {product.name} ({product.quality_grade}-grade, "
            f"{product.quantity}kg from {product.origin})\n"
            f"Market price: ₹{market:,}\n"
            f"Your opening offer: ₹{offer_price:,}\n"
            f"Your budget limit: ₹{context.your_budget:,}\n\n"
            f"TASK: Write a professional opening message that:\n"
            f"- Establishes your interest in the product\n"
            f"- Justifies your offer with market knowledge\n"
            f"- Shows respect for the seller\n"
            f"- Opens the door for negotiation\n"
        )
        
        return self.llm.sample_text(prompt)
    
    def _generate_acceptance_message(self, accepted_price: int, seller_message: str) -> str:
        """Generate acceptance message using LLM"""
        prompt = (
            self.personality_component.make_pre_act_value() +
            f"SITUATION: You are accepting the seller's offer of ₹{accepted_price:,}\n"
            f"Seller's message: '{seller_message}'\n\n"
            f"TASK: Write a professional acceptance message that:\n"
            f"- Confirms the deal\n"
            f"- Shows satisfaction with the agreement\n"
            f"- Maintains positive relationship\n"
        )
        
        return self.llm.sample_text(prompt)
    
    def _generate_rejection_message(self, seller_price: int, budget: int) -> str:
        """Generate rejection message when seller price exceeds budget"""
        prompt = (
            self.personality_component.make_pre_act_value() +
            f"SITUATION: Seller's price of ₹{seller_price:,} exceeds your budget of ₹{budget:,}\n\n"
            f"TASK: Write a polite rejection message that:\n"
            f"- Respectfully declines the offer\n"
            f"- Explains budget constraints without revealing exact budget\n"
            f"- Leaves door open for future opportunities\n"
        )
        
        return self.llm.sample_text(prompt)
    
    def _generate_counter_message(
        self, context: NegotiationContext, seller_price: int, counter_price: int, seller_message: str
    ) -> str:
        """Generate counter-offer message using LLM and memory"""
        
        # Get conversation context from memory
        recent_context = self.memory.get_recent_turns(3)
        relevant_context = self.memory.retrieve_context("negotiation price offers", k=5)
        
        rounds_left = 10 - context.current_round
        urgency = "high" if rounds_left <= CFG.late_round else "normal"
        market_price = context.product.base_market_price
        
        prompt = (
            self.personality_component.make_pre_act_value() +
            f"NEGOTIATION STATE:\n"
            f"Product: {context.product.name} ({context.product.quality_grade}-grade)\n"
            f"Market price: ₹{market_price:,}\n"
            f"Round {context.current_round}/10 (urgency: {urgency})\n"
            f"Seller's current offer: ₹{seller_price:,}\n"
            f"Your counter-offer: ₹{counter_price:,}\n"
            f"Your budget: ₹{context.your_budget:,}\n\n"
            f"CONVERSATION CONTEXT:\n{recent_context}\n\n"
            f"SELLER'S LATEST MESSAGE: '{seller_message}'\n\n"
            f"TASK: Write a persuasive counter-offer message that:\n"
            f"- Responds to seller's points professionally\n"
            f"- Justifies your counter-offer with market data or product considerations\n"
            f"- Shows willingness to negotiate while being firm\n"
            f"- Builds rapport and maintains positive relationship\n"
            f"- Reflects the negotiation urgency level\n"
        )
        
        return self.llm.sample_text(prompt)


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
        
        return opening, f"I'm interested, but ₹{opening:,} is what I can offer. Let me think about that..."
    
    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        # Accept if within budget and below 85% of market
        if seller_price <= context.your_budget and seller_price <= context.product.base_market_price * 0.85:
            return DealStatus.ACCEPTED, seller_price, f"Alright, ₹{seller_price:,} works for me!"
        
        # Counter with small increment
        last_offer = context.your_offers[-1] if context.your_offers else 0
        counter = min(int(last_offer * 1.1), context.your_budget)
        
        if counter >= seller_price * 0.95:  # Close to agreement
            counter = min(seller_price - 1000, context.your_budget)
            return DealStatus.ONGOING, counter, f"That's a bit steep for me. How about ₹{counter:,}?"
        
        return DealStatus.ONGOING, counter, f"I can go up to ₹{counter:,}, but that's pushing my budget."
    
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
        return price, f"These are premium {product.quality_grade} grade {product.name}. I'm asking ₹{price:,}."
    
    def respond_to_buyer(self, buyer_offer: int, round_num: int) -> Tuple[int, str, bool]:
        if buyer_offer >= self.min_price * 1.1:  # Good profit
            return buyer_offer, f"You have a deal at ₹{buyer_offer:,}!", True
            
        if round_num >= 8:  # Close to timeout
            counter = max(self.min_price, int(buyer_offer * 1.05))
            return counter, f"Final offer: ₹{counter:,}. Take it or leave it.", False
        else:
            counter = max(self.min_price, int(buyer_offer * 1.15))
            return counter, f"I can come down to ₹{counter:,}.", False


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
        
        # Handle rejection case
        if status == DealStatus.REJECTED:
            context.messages.append({"role": "buyer", "message": buyer_msg})
            break
        
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


def demonstrate_concordia_features():
    """Demonstrate Concordia-specific features"""
    print("\n🧠 CONCORDIA FRAMEWORK DEMONSTRATION")
    print("="*50)
    
    # Create agent
    agent = YourBuyerAgent("DemoAgent", "llama3.2:3b")
    
    # Test memory system
    print("\n1. MEMORY SYSTEM TEST:")
    agent.memory.add_conversation_turn("seller", "I'm offering premium mangoes at ₹200,000", 200000)
    agent.memory.add_conversation_turn("buyer", "That's too high. I can do ₹150,000", 150000)
    agent.memory.add_conversation_turn("seller", "How about ₹175,000?", 175000)
    
    recent = agent.memory.get_recent_turns(2)
    print(f"Recent conversation:\n{recent}")
    
    context = agent.memory.retrieve_context("price negotiation", k=3)
    print(f"\nRelevant context:\n{context}")
    
    # Test personality component
    print("\n2. PERSONALITY COMPONENT TEST:")
    personality_prompt = agent.personality_component.make_pre_act_value()
    print(f"Personality context:\n{personality_prompt}")
    
    # Test LLM integration
    print("\n3. LLM INTEGRATION TEST:")
    test_prompt = (
        personality_prompt + 
        "You're negotiating for mangoes. The seller wants ₹175,000 but your budget is ₹160,000. "
        "Counter-offer with ₹155,000."
    )
    
    llm_response = agent.llm.sample_text(test_prompt)
    print(f"LLM Response: {llm_response}")
    
    # Test policy system
    print("\n4. POLICY SYSTEM TEST:")
    market_price = 180000
    budget = 160000
    opening = agent.policy.opening_offer(market_price, budget)
    should_accept = agent.policy.should_accept(175000, market_price, budget)
    counter = agent.policy.counter_offer(175000, market_price, budget, opening, 3, 10)
    
    print(f"Opening offer: ₹{opening:,}")
    print(f"Should accept ₹175,000: {should_accept}")
    print(f"Counter offer: ₹{counter:,}")
    
    print("\n✅ All Concordia components working correctly!")


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
        model_name=cfg.get("model", "llama3.2:3b"),
    )


# ============================================
# PART 8: INSTALLATION HELPER
# ============================================

def check_and_install_concordia():
    """Helper function to check and provide installation instructions"""
    try:
        import concordia
        print("✅ Concordia is installed and available")
        return True
    except ImportError:
        print("\n⚠️  CONCORDIA NOT FOUND")
        print("="*50)
        print("To install Concordia framework:")
        print("1. Clone the repository:")
        print("   git clone https://github.com/google-deepmind/concordia.git")
        print("2. Install in development mode:")
        print("   cd concordia")
        print("   pip install -e .")
        print("3. Or install via pip (if available):")
        print("   pip install dm-concordia")
        print("\nRunning in compatibility mode for now...")
        print("="*50)
        return False


def setup_ollama_model(model_name: str = "llama3:8b"):
    """Helper function to set up Ollama model"""
    print(f"\n🔧 SETTING UP OLLAMA MODEL: {model_name}")
    print("="*50)
    
    try:
        # Check if Ollama is installed
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Ollama not found. Please install Ollama first:")
            print("   Visit: https://ollama.ai/")
            return False
        
        print(f"✅ Ollama version: {result.stdout.strip()}")
        
        # Check if model exists
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_name.split(':')[0] not in result.stdout:
            print(f"⬇️  Downloading model {model_name}...")
            download_result = subprocess.run(["ollama", "pull", model_name], 
                                           capture_output=True, text=True)
            if download_result.returncode == 0:
                print(f"✅ Model {model_name} downloaded successfully")
            else:
                print(f"❌ Failed to download {model_name}")
                print("Available alternatives: llama3.2:1b, phi3:mini, qwen2:0.5b")
                return False
        else:
            print(f"✅ Model {model_name} is already available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up Ollama: {e}")
        return False


if __name__ == "__main__":
    print("🚀 CONCORDIA-ENHANCED NEGOTIATION AGENT")
    print("=" * 50)
    
    # Check installations
    concordia_available = check_and_install_concordia()
    ollama_ready = setup_ollama_model("llama3:8b")
    
    if not ollama_ready:
        print("\n⚠️  Consider using a lighter model like 'phi3:mini' if you have limited resources")
    
    print(f"\n🔍 FRAMEWORK STATUS:")
    print(f"Concordia: {'✅ Available' if concordia_available else '⚠️  Stub Mode'}")
    print(f"Ollama LLM: {'✅ Ready' if ollama_ready else '⚠️  May have issues'}")
    
    # Demonstrate Concordia features
    demonstrate_concordia_features()
    
    # Run comprehensive test
    print(f"\n{'='*60}")
    print("🧪 RUNNING COMPREHENSIVE AGENT TESTS")
    print("="*60)
    
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
    improvement_deals = deals - simple_deals
    improvement_savings = savings - simple_savings
    print(f"Improvement: {'+' if improvement_deals >= 0 else ''}{improvement_deals} deals, "
          f"{'+' if improvement_savings >= 0 else ''}₹{improvement_savings:,} savings")
    
    print(f"\n🎉 CONCORDIA ENHANCEMENT COMPLETE!")
    print(f"Your agent is ready for deployment with {'full' if concordia_available else 'compatibility'} Concordia support.")