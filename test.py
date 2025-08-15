from concordia import UnifiedBuyerAgent, Product, NegotiationContext, DealStatus

def run_scenario(name, product, budget, seller_start_multiplier, seller_concede_factor, rounds=10):
    buyer = UnifiedBuyerAgent(name="TestBuyer", model_name="llama3:8b")
    ctx = NegotiationContext(product, budget, 0, [], [], [])

    print(f"\n{'='*100}")
    print(f"TEST SCENARIO: {name}")
    print(f"Product: {product.name} | Market ₹{product.base_market_price:,} | Budget ₹{budget:,}")
    print("=" * 100)

    # Seller first offer
    seller_price = int(product.base_market_price * seller_start_multiplier)
    seller_msg = f"Asking ₹{seller_price:,} ({'high' if seller_start_multiplier > 1 else 'reasonable'} price)"
    ctx.seller_offers.append(seller_price)
    ctx.messages.append({"role": "seller", "message": seller_msg})
    print(f"\nSELLER: {seller_msg}")

    # Buyer opening offer
    offer, msg = buyer.generate_opening_offer(ctx)
    ctx.your_offers.append(offer)
    ctx.messages.append({"role": "buyer", "message": msg})
    print(f"BUYER OFFER: ₹{offer:,} | {msg}")

    # Negotiation loop
    for round_i in range(2, rounds + 1):
        ctx.current_round = round_i
        # Seller concedes based on factor
        seller_price = max(int(product.base_market_price * 0.8), int(seller_price - seller_concede_factor))
        seller_msg = f"Counter offer: ₹{seller_price:,}"
        ctx.seller_offers.append(seller_price)
        ctx.messages.append({"role": "seller", "message": seller_msg})
        print(f"\nSELLER: {seller_msg}")

        status, offer, msg = buyer.respond_to_seller_offer(ctx, seller_price, seller_msg)
        ctx.your_offers.append(offer)
        ctx.messages.append({"role": "buyer", "message": msg})
        print(f"BUYER: ₹{offer:,} | {msg}")

        if status == DealStatus.ACCEPTED:
            print("\n🎉 DEAL ACCEPTED!")
            print(f"Final Price: ₹{offer:,} | Savings: ₹{budget - offer:,}")
            return True
        elif offer >= seller_price:
            print("\n🎉 DEAL MATCHED SELLER PRICE!")
            print(f"Final Price: ₹{offer:,} | Savings: ₹{budget - offer:,}")
            return True

    print("\n❌ DEAL NOT CLOSED")
    return False

def run_all_tests():
    product_easy = Product("Local Bananas", "Fruit", 50, "B", "Tamil Nadu", 50_000)
    product_medium = Product("Premium Alphonso Mangoes", "Fruit", 100, "A", "Ratnagiri", 180_000)
    product_hard = Product("Export Durian", "Fruit", 20, "A+", "Malaysia", 500_000)

    scenarios = [
        # name, product, budget, start price multiplier, seller concede factor
        ("Easy 1", product_easy, 60_000, 1.1, 5_000),
        ("Easy 2", product_easy, 60_000, 1.2, 4_000),
        ("Medium 1", product_medium, 200_000, 1.4, 15_000),
        ("Medium 2", product_medium, 200_000, 1.5, 12_000),
        ("Hard 1", product_hard, 520_000, 1.8, 10_000),
        ("Hard 2", product_hard, 520_000, 2.0, 8_000),
    ]

    results = {}
    for name, product, budget, start_mult, concede in scenarios:
        result = run_scenario(name, product, budget, start_mult, concede)
        results[name] = "✅" if result else "❌"

    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)
    for name, res in results.items():
        print(f"{name}: {res}")

if __name__ == "__main__":
    run_all_tests()
