"""
===========================================
AI NEGOTIATION AGENT - INTERVIEW TEMPLATE
===========================================

Welcome! Your task is to build a BUYER agent that can negotiate effectively
against our hidden SELLER agent. Success is measured by achieving profitable
deals while maintaining character consistency.

INSTRUCTIONS:
1. Read through this entire template first
2. Implement your agent in the marked sections
3. Test using the provided framework
4. Submit your completed code with documentation

"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import random

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
    attributes: Dict[str, Any]

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
        """
        Define your agent's personality traits.
        
        Returns:
            Dict containing:
            - personality_type: str (e.g., "aggressive", "analytical", "diplomatic", "custom")
            - traits: List[str] (e.g., ["impatient", "data-driven", "friendly"])
            - negotiation_style: str (description of approach)
            - catchphrases: List[str] (typical phrases your agent uses)
        """
        pass
    
    @abstractmethod
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        """
        Generate your first offer in the negotiation.
        
        Args:
            context: Current negotiation context
            
        Returns:
            Tuple of (offer_amount, message)
            - offer_amount: Your opening price offer (must be <= budget)
            - message: Your negotiation message (2-3 sentences, include personality)
        """
        pass
    
    @abstractmethod
    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        """
        Respond to the seller's offer.
        
        Args:
            context: Current negotiation context
            seller_price: The seller's current price offer
            seller_message: The seller's message
            
        Returns:
            Tuple of (deal_status, counter_offer, message)
            - deal_status: ACCEPTED if you take the deal, ONGOING if negotiating
            - counter_offer: Your counter price (ignored if deal_status is ACCEPTED)
            - message: Your response message
        """
        pass
    
    @abstractmethod
    def get_personality_prompt(self) -> str:
        """
        Return a prompt that describes how your agent should communicate.
        This will be used to evaluate character consistency.
        
        Returns:
            A detailed prompt describing your agent's communication style
        """
        pass


# ============================================
# PART 3: YOUR IMPLEMENTATION STARTS HERE
# ============================================

class YourBuyerAgent(BaseBuyerAgent):
    """
    YOUR BUYER AGENT IMPLEMENTATION
    
    Tips for success:
    1. Stay in character - consistency matters!
    2. Never exceed your budget
    3. Aim for the best deal, but know when to close
    4. Use market price as reference
    5. Maximum 10 rounds - don't let negotiations timeout
    """
    
    def define_personality(self) -> Dict[str, Any]:
        """
        TODO: Define your agent's unique personality
        
        Choose from: aggressive, analytical, diplomatic, or create custom
        """
        # IMPLEMENT YOUR PERSONALITY HERE
        return {
            "personality_type": "custom",  # Change this
            "traits": ["trait1", "trait2"],  # Define 3-5 traits
            "negotiation_style": "Description of your approach",
            "catchphrases": ["phrase1", "phrase2"]  # 2-3 signature phrases
        }
    
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        """
        TODO: Generate your opening offer
        
        Consider:
        - Product's base market price
        - Your budget constraints  
        - Your personality's approach
        - Quality grade and origin
        """
        # IMPLEMENT YOUR OPENING STRATEGY HERE
        
        # Example (replace with your logic):
        opening_price = int(context.product.base_market_price * 0.7)  # 30% below market
        
        # Ensure within budget
        opening_price = min(opening_price, context.your_budget)
        
        message = f"I'll offer ₹{opening_price} for these {context.product.name}."
        
        return opening_price, message
    
    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        """
        TODO: Implement your response strategy
        
        Remember:
        - Analyze if the seller's price is acceptable
        - Consider how many rounds have passed
        - Maintain your personality
        - Know when to accept vs counter-offer
        """
        # IMPLEMENT YOUR RESPONSE STRATEGY HERE
        
        # Example logic (replace with your strategy):
        
        # Check if price is within budget and reasonable
        if seller_price <= context.your_budget:
            # Simple acceptance logic - improve this!
            if seller_price <= context.product.base_market_price * 0.9:
                return DealStatus.ACCEPTED, seller_price, f"Deal accepted at ₹{seller_price}!"
        
        # Counter-offer logic
        if context.current_round >= 8:  # Getting close to timeout
            # Make more aggressive moves
            counter_offer = min(int(seller_price * 0.95), context.your_budget)
        else:
            # Normal negotiation
            counter_offer = min(int(seller_price * 0.85), context.your_budget)
        
        message = f"I can go up to ₹{counter_offer}."
        
        return DealStatus.ONGOING, counter_offer, message
    
    def get_personality_prompt(self) -> str:
        """
        TODO: Write a detailed prompt for your agent's communication style
        
        This should be 3-5 sentences describing how your agent talks,
        what phrases they use, their tone, etc.
        """
        # IMPLEMENT YOUR PERSONALITY PROMPT HERE
        return """
        I am a [your personality type] buyer who [describe communication style].
        I typically [describe behavior patterns].
        My catchphrases include [list some phrases].
        """

    # ============================================
    # OPTIONAL: Add helper methods below
    # ============================================
    
    def analyze_negotiation_progress(self, context: NegotiationContext) -> Dict[str, Any]:
        """Optional: Analyze how the negotiation is progressing"""
        # Add your analysis logic
        pass
    
    def calculate_fair_price(self, product: Product) -> int:
        """Optional: Calculate what you consider a fair price"""
        # Add your pricing logic
        pass


# ============================================
# PART 4: EXAMPLE SIMPLE AGENT (FOR REFERENCE)
# ============================================

class ExampleSimpleAgent(BaseBuyerAgent):
    """
    A simple example agent that you can use as reference.
    This agent has basic logic - you should do better!
    """
    
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
    your_agent = YourBuyerAgent("TestBuyer")
    
    print("="*60)
    print(f"TESTING YOUR AGENT: {your_agent.name}")
    print(f"Personality: {your_agent.personality['personality_type']}")
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
            
            print(f"\nTest: {product.name} - {scenario} scenario")
            print(f"Your Budget: ₹{buyer_budget:,} | Market Price: ₹{product.base_market_price:,}")
            
            result = run_negotiation_test(your_agent, product, buyer_budget, seller_min)
            
            if result["deal_made"]:
                deals_made += 1
                total_savings += result["savings"]
                print(f"✅ DEAL at ₹{result['final_price']:,} in {result['rounds']} rounds")
                print(f"   Savings: ₹{result['savings']:,} ({result['savings_pct']:.1f}%)")
                print(f"   Below Market: {result['below_market_pct']:.1f}%")
            else:
                print(f"❌ NO DEAL after {result['rounds']} rounds")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print(f"Deals Completed: {deals_made}/6")
    print(f"Total Savings: ₹{total_savings:,}")
    print(f"Success Rate: {deals_made/6*100:.1f}%")
    print("="*60)


# ============================================
# PART 7: EVALUATION CRITERIA
# ============================================

"""
YOUR SUBMISSION WILL BE EVALUATED ON:

1. **Deal Success Rate (30%)**
   - How often you successfully close deals
   - Avoiding timeouts and failed negotiations

2. **Savings Achieved (30%)**
   - Average discount from seller's opening price
   - Performance relative to market price

3. **Character Consistency (20%)**
   - How well you maintain your chosen personality
   - Appropriate use of catchphrases and style

4. **Code Quality (20%)**
   - Clean, well-structured implementation
   - Good use of helper methods
   - Clear documentation

BONUS POINTS FOR:
- Creative, unique personalities
- Sophisticated negotiation strategies
- Excellent adaptation to different scenarios
"""

# ============================================
# PART 8: SUBMISSION CHECKLIST
# ============================================

"""
BEFORE SUBMITTING, ENSURE:

[ ] Your agent is fully implemented in YourBuyerAgent class
[ ] You've defined a clear, consistent personality
[ ] Your agent NEVER exceeds its budget
[ ] You've tested using test_your_agent()
[ ] You've added helpful comments explaining your strategy
[ ] You've included any additional helper methods

SUBMIT:
1. This completed template file
2. A 1-page document explaining:
   - Your chosen personality and why
   - Your negotiation strategy
   - Key insights from testing

FILENAME: negotiation_agent_[your_name].py
"""

if __name__ == "__main__":
    # Run this to test your implementation
    test_your_agent()
    
    # Uncomment to see how the example agent performs
    # print("\n\nTESTING EXAMPLE AGENT FOR COMPARISON:")
    # example_agent = ExampleSimpleAgent("ExampleBuyer")
    # test_your_agent()
