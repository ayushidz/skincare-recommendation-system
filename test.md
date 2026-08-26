# Test Cases

## Successful Scenarios

### Test Case 1 — Oily Skin + Acne

**Input**
- Skin Type: Oily
- Concern: Acne
- Product Category: All
- Recommendations: 5

**Expected Result**

The system should prioritize products suitable for oily skin and targeting acne.

**Why this is a successful scenario**

Both skin type and concern are directly represented in the product metadata, giving the recommendation model strong signals for ranking relevant products.

---

### Test Case 2 — Dry Skin + Hydration

**Input**
- Skin Type: Dry
- Concern: Hydration
- Product Category: All
- Recommendations: 5

**Expected Result**

Products associated with dry skin and hydration should appear near the top of the recommendations.

**Why this is a successful scenario**

The requested attributes directly correspond to product metadata available in the dataset, allowing both explicit matching and TF-IDF similarity to contribute to the ranking.

---

### Test Case 3 — Oily Skin + Multiple Concerns

**Input**
- Skin Type: Oily
- Concerns: Acne, Dark Spots
- Product Category: All
- Recommendations: 5

**Expected Result**

Products matching both concerns should generally rank higher than products matching only one concern.

**Why this is a successful scenario**

This tests the system's ability to handle multiple simultaneous user preferences. The concern score rewards products that overlap with a larger proportion of the user's selected concerns.

---

# Failure Scenarios

### Test Case 4 — Sensitive Skin + Anti-Aging + Dark Spots

**Input**
- Skin Type: Sensitive
- Concerns: Anti-Aging, Dark Spots
- Product Category: All
- Recommendations: 5

**Expected Result**

The system should return the closest available products even if no product perfectly matches the complete profile.

**Observed Limitation**

Some combinations of skin type and concerns have limited representation in the dataset, so recommendations may match only part of the user's profile.

**Why the system struggles**

The model can only rank products that exist in the available catalog. It cannot recommend products for combinations that are not represented in the dataset.

---

### Test Case 5 — Oily Skin + Acne + Eye Care

**Input**
- Skin Type: Oily
- Concern: Acne
- Product Category: Eye Care
- Recommendations: 5

**Expected Result**

The system should prioritize Eye Care products while considering the user's skin type and concern.

**Observed Limitation**

The candidate pool can become very small or contain weaker matches when the user applies highly specific filters.

**Why the system struggles**

The dataset contains only 214 products and does not provide complete coverage for every combination of skin type, concern, and product category.

---

# Summary

| Scenario | Result |
|---|---|
| Oily + Acne | Successful |
| Dry + Hydration | Successful |
| Oily + Acne + Dark Spots | Successful |
| Sensitive + Anti-Aging + Dark Spots | Imperfect |
| Oily + Acne + Eye Care | Imperfect |

The failure scenarios are intentional. They demonstrate the limitations of a small content-based catalog and identify areas where additional product data and user interaction data would improve recommendation quality.