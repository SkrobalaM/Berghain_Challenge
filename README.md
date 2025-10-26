# My Berghain Challenge Journey: From Simple Heuristics to Mathematical Optimization

## The Challenge: Being a Nightclub Bouncer

Imagine you're the bouncer at Berghain, Berlin's most exclusive nightclub. Your job is to fill the venue with exactly 1000 people, but there's a catch - you need to meet specific requirements. For example:
- At least 40% must be Berlin locals
- At least 80% must be wearing all black
- At least 30% must be creative types

People arrive one by one, and you have to decide immediately whether to let them in or turn them away. You can't see who's coming next, and if you reject more than 20,000 people, you lose the game. Your goal? Fill the venue with as few rejections as possible while meeting all the requirements.

This isn't just a game - it's a complex optimization problem that I spent days trying to solve, going through multiple failed approaches before finding a good solution.

Let me explain how I managed to get rank 68/1330 on this challenge.

## My First Attempt: Rarity Scoring

My first approach was to prioritize people who had traits I was still missing. I thought: "Let me give each person a score based on how rare and needed their traits are."

**What I did:**
- Calculated a "rarity score" for each person based on what traits I still needed
- Accepted people above a certain score threshold
- The rarer the trait, the higher the score

**Why it failed:**
I was still thinking too simplistically. I was using a fixed threshold, which meant I'd accept everyone above a certain score and reject everyone below it.

**The problem:** I was treating this like a simple ranking problem, but it's actually a complex resource allocation problem where the optimal strategy changes based on your current situation. The score computation is a hard-coded function that needs to be fine-tuned, and finding a good one is like having a good recipe.

## My Second Attempt: The "Phase" Strategy

My next approach was pretty naive. I thought: "Let me be super picky at first, then gradually get more lenient as the night goes on and the quotas are met."

**What I did:**
- **Phase 1**: Only accept people with 3+ good traits
- **Phase 2**: Accept people with 2+ traits, plus some random people
- **Phase 3**: Accept anyone with 1+ trait, plus more random people
- **Phase 4**: Accept everyone

**Why it failed:**
This was like being a terrible bouncer who doesn't understand the crowd. I was rejecting perfectly good people early on, then desperately accepting anyone later. It didn't work because I wasn't thinking about what I actually needed.

**The problem:** I was making decisions based on gut feeling instead of math. I wasn't considering that some traits are rarer than others, or that traits can be correlated (like how creative people might also be more likely to be international).

## My Third Attempt: Simple Probability

For the next try, I focused on just two traits: "young" and "well-dressed." I calculated the exact probabilities and used those to make decisions.

**What I did:**
- Calculated how often people have both traits together
- Used these probabilities to decide who to let in
- Added a small random factor to allow the venue to be filled faster

**Why it failed:**
This worked okay for just two traits. In the first scenario it was easy to understand the relationship between the 2 traits and see that some combinations of traits are more complicated to have than others. But scenarios 2 and 3 have more traits with complex relationships, and having a human understanding of people distribution is too complicated. With 6 different traits in scenario 3, you have 2^6=64 different kinds of people.

**The problem:** I was oversimplifying. Real people have multiple attributes that interact in complicated ways, and I needed a method that could handle this complexity for future scenarios.

## The Breakthrough: Mathematical Optimization

After these failures, I realized I needed to think about this completely differently. Instead of using heuristics or simple rules, I needed to use mathematics to find the truly optimal strategy.

### The Key Insight

The breakthrough came when I realized this is actually a **linear programming problem**. Think of it like this:

- You have a limited resource (1000 spots in the club)
- You have multiple requirements to meet (minimum counts for each trait)
- You want to minimize waste (rejections)
- You need to make decisions in real-time

The solution is to calculate, for each possible type of person, the optimal probability of accepting them based on your current needs.

### How It Works (In Simple Terms)

1. **Model the Crowd**: First, I built a mathematical model that understands how traits are related. For example, if someone is creative, they're more likely to also be international.

2. **Calculate What You Need**: At any moment, I calculate exactly how many people with each trait I still need to meet the requirements.

3. **Find the Optimal Strategy**: Using linear programming, I calculate the best acceptance probability for each type of person. This tells me: "Given my current situation, I should accept 80% of a specific kind of person like a creative + local + well connected."

4. **Make Real-Time Decisions**: When someone arrives, I look at their traits, check my optimal probabilities, and make a decision.

5. **Update Continuously**: After each decision, I recalculate everything based on my new situation.

### Why This Works

This approach works because it's mathematically optimal. Instead of guessing or using rules of thumb, it calculates the exact best strategy for any situation.

The beauty is that it adapts in real-time. If you're running low on creative people, it automatically becomes more likely to accept creative people. If you have plenty of techno lovers, it becomes more selective about them.

## Making It Work: Optimization Strategies

Once I had the mathematical solution, I faced some practical challenges that I needed to solve to actually compete effectively:

### Building a Simulator
Because every decision requires an API call, testing my solution was painfully slow. I couldn't afford to test hundreds of variations by calling the real API. So I built a simulator that mimics people arriving according to the exact probabilities and correlations from the challenge data. This let me test and refine my approach thousands of times before making a single real API call.

### Maximizing My Chances
The scoring system only keeps your best attempt for each scenario, and your final score is the sum of the lowest rejections across all three scenarios. To maximize my chances of getting good runs, I:

- **Ran multiple scenarios simultaneously**: The challenge allowed 15 concurrent attempts, so I ran as many as possible at the same time
- **Used cloud computing**: My home internet was unreliable, so I deployed my solution to Railway (a cloud platform) to run 24/7 for the final two days
- **Automated everything**: The system would automatically restart failed attempts and keep trying until it got good results

This strategy paid off - I went from rank 200 to rank 68 in the final days by running thousands of attempts and keeping only the best results.

## The Technical Implementation

For those interested in the mathematical details, here's how the final solution works:

### 1. Gaussian Copula Modeling for Joint Distribution

The solution models the joint distribution of all attributes using a Gaussian copula approach. This is crucial because attributes are correlated (e.g., Berlin locals are more likely to be not well connected).

**Mathematical Process:**
1. **Convert correlations to latent correlations**: The given correlation matrix `Phi` contains Pearson correlations between binary attributes. These are converted to latent correlations `R` for the underlying multivariate normal distribution using the `invert_phi_to_rho()` function.

2. **Generate samples**: Using the latent correlation matrix `R`, the code generates 200,000 samples from a multivariate normal distribution:
   ```python
   Z = rng.standard_normal(size=(n_samples, n)) @ L.T
   X = (Z > t).astype(np.int8)  # Convert to binary attributes
   ```

3. **Estimate joint probabilities**: The samples are used to estimate the exact joint probability `pi` for each of the 2^n possible attribute combinations.

### 2. Linear Programming Optimization

The core innovation is using linear programming to find optimal acceptance probabilities for each person type in real-time.

**Mathematical Formulation:**
- **Variables**: `y_s` = acceptance probability for person type `s` (where `s` represents a specific combination of attributes)
- **Objective**: Maximize total acceptance probability: `maximize Σ y_s`
- **Constraints**: For each attribute `j`, ensure we meet the minimum requirement:
  ```
  Σ (y_s * indicator_s_j) ≥ θ_j * remaining_slots
  ```
  where `indicator_s_j = 1` if person type `s` has attribute `j`, and `θ_j = targets_remaining[j] / slots_remaining`

- **Bounds**: `0 ≤ y_s ≤ π_s` (can't accept more than the natural frequency)

**Implementation:**
```python
def build_indicator(ATTRS, pi, targets_remaining, slots_remaining):
    theta = np.array([targets_remaining[a]/slots_remaining for a in ATTRS])
    # Set up constraint matrix A_ub and bounds
    res = linprog(-np.ones(n_states), A_ub=A_ub, b_ub=b_ub, 
                  bounds=[(0.0, float(pi_s)) for pi_s in pi], method="highs")
    # Convert solution to acceptance probabilities
    accept_prob = np.divide(y, pi, out=np.zeros_like(y), where=pi>1e-15)
```

### 3. Real-Time Decision Making

When a person arrives, the system:
1. **Identifies person type**: Converts their attributes to a bit pattern
2. **Looks up probability**: Retrieves the optimal acceptance probability for that type
3. **Makes decision**: Uses `rng.random() < p` to decide accept/reject
4. **Updates strategy**: Recalculates optimal probabilities after each decision

### 4. Dynamic Adaptation

The system continuously adapts by:
- **Recalculating targets**: `new_targets = max(targets[trait] - berg[trait], 0)`
- **Updating probabilities**: Re-running the LP optimization with new constraints
- **Handling edge cases**: When quotas are met, the system becomes more selective

### Key Mathematical Functions

```python
# Convert correlation matrix to latent correlation matrix
invert_phi_to_rho(phi_target, p1, p2)

# Generate person with correlated attributes using multivariate normal
generate_person(p_vec, latent_corr, rng)

# Solve LP to find optimal acceptance probabilities
build_indicator(ATTRS, pi, targets_remaining, slots_remaining)

# Make probabilistic accept/reject decision
indicator(indicator_data, bits, rng, done)
```

### Why This Works Mathematically

1. **Optimality**: Linear programming guarantees the mathematically optimal solution for any given state
2. **Feasibility**: The LP constraints ensure all requirements can be met if mathematically possible
3. **Adaptivity**: The solution updates in real-time as conditions change
4. **Scalability**: Works for any number of attributes (tested up to 6 attributes = 64 person types)

## The Results

The final solution achieves near-perfect performance:
- **Success Rate**: Nearly 100% for all feasible scenarios
- **Efficiency**: Minimizes rejections through optimal resource allocation
- **Adaptability**: Handles any number of attributes and complex correlations
- **Scalability**: Works for scenarios with 2-6+ different traits
