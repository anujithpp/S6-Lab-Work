from itertools import combinations

# ---------------- TRANSACTIONS ----------------
transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Butter', 'Eggs']
]

N = len(transactions)

# ---------------- SUPPORT FUNCTION ----------------
def support(itemset, transactions):
    count = 0
    for t in transactions:
        if set(itemset).issubset(t):
            count += 1
    return count / len(transactions)

# ---------------- GENERATE CANDIDATES ----------------
def generate_candidates(items, k):
    return list(combinations(items, k))

# ---------------- APRIORI (GENERALIZED) ----------------
def apriori(transactions, min_support):
    all_items = sorted(set(item for t in transactions for item in t))
    frequent_itemsets = {}

    k = 1
    current_items = all_items

    while True:
        candidates = generate_candidates(current_items, k)

        frequent_k = []
        print(f"\nCandidate {k}-itemsets:", candidates)

        for itemset in candidates:
            sup = support(itemset, transactions)
            if sup >= min_support:
                frequent_k.append(itemset)
                frequent_itemsets[itemset] = sup

        if not frequent_k:
            break

        print(f"\nFrequent {k}-itemsets:")
        for f in frequent_k:
            print(f, "Support:", round(frequent_itemsets[f], 2))

        # Prepare items for next iteration
        current_items = sorted(set(item for itemset in frequent_k for item in itemset))
        k += 1

    return frequent_itemsets

# ---------------- ASSOCIATION RULES ----------------
def generate_rules(frequent_itemsets, min_confidence):
    print("\n============================")
    print("Association Rules Generated")
    print("============================")

    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue

        itemset_support = frequent_itemsets[itemset]

        # Generate rules: A -> B
        for i in range(1, len(itemset)):
            for A in combinations(itemset, i):
                B = tuple(set(itemset) - set(A))

                sup_A = frequent_itemsets.get(tuple(sorted(A)), support(A, transactions))
                sup_B = frequent_itemsets.get(tuple(sorted(B)), support(B, transactions))

                confidence = itemset_support / sup_A
                lift = confidence / sup_B

                if confidence >= min_confidence:
                    print(f"{A} -> {B} | Support={itemset_support:.2f}, "
                          f"Confidence={confidence:.2f}, Lift={lift:.2f}")

# ---------------- MAIN PROGRAM ----------------
min_support = 0.4
min_confidence = 0.6

print("Transactions:")
for t in transactions:
    print(t)

# Run Apriori
frequent_itemsets = apriori(transactions, min_support)

print("\n============================")
print("All Frequent Itemsets Found")
print("============================")
for itemset, sup in frequent_itemsets.items():
    print(itemset, "Support:", round(sup, 2))

# Generate association rules
generate_rules(frequent_itemsets, min_confidence)
