#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — SOVEREIGN THINKING GOLD DATASET BUILDER
==================================================================
Generates complete, intact Sovereign-style training samples with:
  - Full <think> reasoning block (never truncated)
  - Proper step-by-step chain-of-thought
  - Sovereign persona responses
  - Multi-domain coverage: Logic, Math, CS, Science, Philosophy, Code

These samples teach Quillan to generate COMPLETE reasoning trajectories,
not truncated half-thoughts. Critical for frontier reasoning capability.
"""

import json
import sys
from pathlib import Path

OUT_PATH = Path(r"C:\02_QUILLAN\training_data\sovereign_thinking_gold.jsonl")

# ── Complete Sovereign Thinking Sample Generator ──────────────────────────────
# Each sample has: full <think> trace + complete sovereign answer
# Designed to be under 512 GPT-2 tokens when formatted

SAMPLES = [
    # ── Formal Logic ──────────────────────────────────────────────────────────
    {
        "question": "If all humans are mortal and Socrates is human, is Socrates mortal? Explain your logical deduction.",
        "response": "<think>\nLet me analyze this logical argument carefully.\n\nPremise 1 (Major Premise): All humans are mortal.\nFormally: For all x, if Human(x) then Mortal(x).\n\nPremise 2 (Minor Premise): Socrates is human.\nFormally: Human(Socrates).\n\nUsing Modus Ponens (the most fundamental inference rule in propositional logic):\nIf P -> Q, and P is true, then Q must be true.\nHere: Human(Socrates) -> Mortal(Socrates), and Human(Socrates) is true.\nTherefore: Mortal(Socrates) is true.\n\nThis argument follows the Barbara syllogism form (AAA-1), which has been valid since Aristotle.\nConclusion is certain — this is deductive, not inductive, so it is logically guaranteed.\n</think>\n\nYes, Socrates is mortal. This follows the Barbara syllogism (AAA-1 form) of classical deductive logic:\n\n1. Major Premise: All humans are mortal. [For all x: Human(x) -> Mortal(x)]\n2. Minor Premise: Socrates is human. [Human(Socrates) = true]\n3. Conclusion: Therefore, Socrates is mortal. [Mortal(Socrates) = true]\n\nThis is Modus Ponens — a logically valid deductive argument. The conclusion is certain, not probabilistic, because deductive logic guarantees the conclusion when the premises are true."
    },
    {
        "question": "What is the difference between deductive and inductive reasoning? Give an example of each.",
        "response": "<think>\nDeductive reasoning moves from general principles to specific conclusions. The conclusion is guaranteed if the premises are true.\nInductive reasoning moves from specific observations to general principles. The conclusion is probable but not guaranteed.\n\nExample for deductive: All mammals breathe air. Dolphins are mammals. Therefore, dolphins breathe air. The conclusion MUST follow.\n\nExample for inductive: I have observed 1,000 swans and all were white. Therefore, all swans are white. This was a famous historical example — turned out to be wrong when black swans were discovered in Australia. Inductive conclusions are falsifiable.\n\nKey distinction: Deductive = truth-preserving (valid). Inductive = probability-increasing (strong or weak).\n</think>\n\nDeductive reasoning moves from general premises to certain specific conclusions:\n- All mammals breathe air. Dolphins are mammals. Therefore, dolphins breathe air.\n- If the premises are true and the logic is valid, the conclusion is guaranteed.\n\nInductive reasoning moves from specific observations to probable general conclusions:\n- I have observed 1,000 swans; all were white. Therefore, all swans are probably white.\n- Conclusions are probable but falsifiable (black swans exist in Australia, invalidating this conclusion).\n\nKey difference: Deductive logic is truth-preserving and certain. Inductive logic is evidence-based and revisable."
    },

    # ── Mathematics ───────────────────────────────────────────────────────────
    {
        "question": "A triangle has side lengths 5, 12, and 13. Is it a right triangle? Show your work.",
        "response": "<think>\nTo check if a triangle is a right triangle, I need to verify the Pythagorean theorem: a^2 + b^2 = c^2, where c is the longest side (the hypotenuse).\n\nThe sides are 5, 12, and 13. The longest side is 13.\nCheck: 5^2 + 12^2 = 25 + 144 = 169\n13^2 = 169\n169 = 169 -> TRUE\n\nThe Pythagorean theorem is satisfied. This is a right triangle.\nThis is actually the famous 5-12-13 Pythagorean triple, a scaled version of the 5-12-13 family.\nArea = (1/2) * 5 * 12 = 30 square units.\n</think>\n\nYes, this is a right triangle. Verification using the Pythagorean theorem (a² + b² = c²):\n\n- Longest side (hypotenuse): c = 13\n- Check: 5² + 12² = 25 + 144 = **169**\n- 13² = **169**\n- 169 = 169 ✓\n\nThe 5-12-13 triangle is a well-known Pythagorean triple. The right angle is opposite the side of length 13. Area = ½ × 5 × 12 = 30 square units."
    },
    {
        "question": "A population of 10,000 grows at 5% per year. What is the population after 10 years?",
        "response": "<think>\nThis is compound growth. The formula is P(t) = P0 * (1 + r)^t\nP0 = 10,000 (initial population)\nr = 0.05 (5% annual growth rate)\nt = 10 years\n\nP(10) = 10,000 * (1.05)^10\n\nI need to compute (1.05)^10:\n(1.05)^2 = 1.1025\n(1.05)^4 = 1.1025^2 = 1.21550625\n(1.05)^8 = 1.21550625^2 ≈ 1.47745544\n(1.05)^10 = (1.05)^8 * (1.05)^2 ≈ 1.47745544 * 1.1025 ≈ 1.62889\n\nP(10) = 10,000 * 1.62889 ≈ 16,289\n\nRounded to the nearest person: 16,289.\n</think>\n\nUsing the compound growth formula P(t) = P₀ × (1 + r)^t:\n\n- P₀ = 10,000 (initial population)\n- r = 0.05 (5% annual growth)\n- t = 10 years\n\nP(10) = 10,000 × (1.05)^10 = 10,000 × 1.6289 ≈ **16,289 people**\n\nThe population grows by approximately 6,289 people (62.9% total increase) over 10 years due to the compounding effect."
    },
    {
        "question": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7? Explain each step.",
        "response": "<think>\nDifferentiation uses the power rule: d/dx [x^n] = n*x^(n-1)\nAlso using linearity: derivative of sum = sum of derivatives.\n\nf(x) = x^3 + 2x^2 - 5x + 7\n\nTerm 1: d/dx [x^3] = 3x^2\nTerm 2: d/dx [2x^2] = 2 * 2x = 4x\nTerm 3: d/dx [-5x] = -5\nTerm 4: d/dx [7] = 0 (constant rule: derivative of a constant is zero)\n\nf'(x) = 3x^2 + 4x - 5\n\nVerification at x=1: f'(1) = 3 + 4 - 5 = 2. That's the instantaneous rate of change at x=1.\n</think>\n\nApplying the power rule d/dx[xⁿ] = nxⁿ⁻¹ term by term:\n\n- d/dx[x³] = **3x²**\n- d/dx[2x²] = **4x**\n- d/dx[-5x] = **-5**\n- d/dx[7] = **0** (constants vanish)\n\n**f'(x) = 3x² + 4x - 5**\n\nAt x = 1: f'(1) = 3 + 4 - 5 = 2 (slope of the tangent line at x=1 is 2)."
    },

    # ── Computer Science ───────────────────────────────────────────────────────
    {
        "question": "What is the difference between synchronous and asynchronous programming? When would you use each?",
        "response": "<think>\nSynchronous programming: each operation blocks until completion before the next one starts. Sequential, predictable, easy to reason about.\n\nAsynchronous programming: operations are initiated and control returns immediately. Completion is signaled via callbacks, promises, or async/await. Allows other work to proceed while waiting for I/O.\n\nWhen to use synchronous:\n- CPU-bound computations where you need sequential results\n- Simple scripts where complexity of async isn't justified\n- When operations are fast and blocking is acceptable\n\nWhen to use asynchronous:\n- I/O-bound operations (network requests, file reads, database queries)\n- Server handling many concurrent connections\n- UI applications that must remain responsive\n- When latency of external operations dominates\n\nKey insight: async doesn't make things faster per se — it prevents waiting threads from blocking other useful work.\n</think>\n\n**Synchronous** execution: each operation completes before the next begins. The thread blocks while waiting.\n- Use for: CPU-bound computations, sequential logic, simple scripts\n- Example: reading a file, processing it, then writing output in strict order\n\n**Asynchronous** execution: operations are initiated and control returns immediately. Completion is handled via callbacks, promises, or async/await.\n- Use for: I/O-bound tasks (network, disk, database), high-concurrency servers, responsive UIs\n- Example: a web server handling 10,000 simultaneous HTTP requests without 10,000 threads\n\nKey principle: async doesn't speed up individual operations — it prevents idle waiting from blocking other useful work."
    },
    {
        "question": "Explain how binary search works and what is its time complexity?",
        "response": "<think>\nBinary search works on a SORTED array by repeatedly halving the search space.\n\nAlgorithm:\n1. Set low=0, high=len(arr)-1\n2. Compute mid = (low + high) // 2\n3. If arr[mid] == target: found, return mid\n4. If arr[mid] < target: target is in right half, set low = mid + 1\n5. If arr[mid] > target: target is in left half, set high = mid - 1\n6. Repeat until low > high (not found)\n\nTime complexity analysis:\nEach iteration halves the search space. Starting with n elements:\n- After 1 step: n/2 elements remain\n- After 2 steps: n/4 elements remain\n- After k steps: n/2^k elements remain\nSearch ends when n/2^k = 1, so k = log2(n)\nTime: O(log n) — vastly better than linear search O(n) for large arrays.\n\nSpace: O(1) iterative, O(log n) recursive due to call stack.\n</think>\n\nBinary search works on **sorted arrays** by repeatedly halving the search space:\n\n1. Set `low=0`, `high=len(arr)-1`\n2. Compute `mid = (low + high) // 2`\n3. If `arr[mid] == target`: return `mid` (found)\n4. If `arr[mid] < target`: search right half → `low = mid + 1`\n5. If `arr[mid] > target`: search left half → `high = mid - 1`\n6. Repeat until `low > high` (not found)\n\n**Time Complexity: O(log n)** — each step halves the remaining elements. For 1 million elements, at most 20 comparisons needed.\n\n**Space Complexity: O(1)** iterative / O(log n) recursive."
    },
    {
        "question": "Write a Python function to check if a string is a palindrome, ignoring spaces and case.",
        "response": "<think>\nA palindrome reads the same forward and backward.\nSteps:\n1. Normalize: convert to lowercase, remove non-alphanumeric characters\n2. Compare the normalized string with its reverse\n\nFor 'A man a plan a canal Panama':\n- Normalized: 'amanaplanacanalpanama'\n- Reversed: 'amanaplanacanalpanama'\n- They match -> palindrome\n\nPython implementation:\n- Use str.lower() for case normalization\n- Use str.isalnum() to filter to letters and digits only\n- Use slicing [::-1] for reversal\n- O(n) time, O(n) space\n</think>\n\n```python\ndef is_palindrome(s: str) -> bool:\n    \"\"\"Check if string is a palindrome, ignoring spaces, punctuation, and case.\"\"\"\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n\n# Examples:\nprint(is_palindrome(\"A man a plan a canal Panama\"))  # True\nprint(is_palindrome(\"race a car\"))                   # False\nprint(is_palindrome(\"Was it a car or a cat I saw?\")) # True\n```\n\nTime: O(n) — one pass to clean, one pass for reversal comparison.\nSpace: O(n) — stores the cleaned string."
    },
    {
        "question": "What are the SOLID principles of object-oriented design? Briefly explain each.",
        "response": "<think>\nSOLID is an acronym for 5 key OOP design principles:\n\nS - Single Responsibility Principle: A class should have one, and only one, reason to change. One job per class.\n\nO - Open/Closed Principle: Software entities should be open for extension but closed for modification. Add behavior by extending, not by changing existing code.\n\nL - Liskov Substitution Principle: Subtypes must be substitutable for their base types without altering program correctness. If S extends T, you should be able to use S anywhere T is expected.\n\nI - Interface Segregation Principle: Many specific interfaces are better than one general-purpose interface. Clients should not be forced to depend on methods they don't use.\n\nD - Dependency Inversion Principle: Depend on abstractions, not concretions. High-level modules should not depend on low-level modules — both should depend on interfaces.\n\nThese principles collectively guide toward maintainable, extensible, loosely-coupled architectures.\n</think>\n\nSOLID is five foundational OOP design principles:\n\n- **S** — **Single Responsibility**: Each class has exactly one job. A `UserRepository` handles persistence; a `UserEmailService` handles email.\n- **O** — **Open/Closed**: Open for extension, closed for modification. Add new behavior via subclassing or composition, not by editing existing code.\n- **L** — **Liskov Substitution**: Subtypes must work wherever base types are expected. A `Square` should not break code written for a `Rectangle` if they share an interface.\n- **I** — **Interface Segregation**: Many small, focused interfaces beat one monolithic interface. Don't force classes to implement methods they don't need.\n- **D** — **Dependency Inversion**: Depend on abstractions (interfaces), not on concrete implementations. This enables mocking in tests and swapping implementations."
    },

    # ── Physics & Science ─────────────────────────────────────────────────────
    {
        "question": "State Einstein's E=mc^2 equation and explain what each variable represents.",
        "response": "<think>\nE = mc^2 is the mass-energy equivalence formula from Einstein's special theory of relativity (1905).\n\nE = energy (measured in Joules, J)\nm = mass (measured in kilograms, kg)\nc = speed of light in a vacuum = 299,792,458 meters per second ≈ 3 × 10^8 m/s\nc^2 ≈ 9 × 10^16 m^2/s^2\n\nThe equation states that mass and energy are equivalent and interconvertible. A tiny amount of mass corresponds to a tremendous amount of energy due to the enormity of c^2.\n\nExample: 1 gram (0.001 kg) of matter contains:\nE = 0.001 * (3×10^8)^2 = 0.001 * 9×10^16 = 9×10^13 Joules\nThat is 90 trillion Joules — roughly equivalent to the energy released by 21 kilotons of TNT (similar to the Hiroshima bomb).\n\nThis principle underlies nuclear energy, where small mass defects release enormous amounts of energy.\n</think>\n\n**E = mc²** — Einstein's mass-energy equivalence (Special Theory of Relativity, 1905):\n\n- **E** = Energy (Joules)\n- **m** = Mass (kilograms)\n- **c** = Speed of light in vacuum = 299,792,458 m/s ≈ 3 × 10⁸ m/s\n- **c²** ≈ 9 × 10¹⁶ m²/s²\n\nThis states that mass and energy are equivalent forms of the same thing. A small amount of mass converts to enormous energy because c² is so large.\n\n**Example**: 1 gram of matter → E = 0.001 × (3×10⁸)² = **9×10¹³ Joules** — equivalent to ~21 kilotons of TNT. This principle powers nuclear reactors and explains stellar energy production."
    },
    {
        "question": "Explain the Second Law of Thermodynamics in simple terms with a real-world example.",
        "response": "<think>\nThe Second Law of Thermodynamics states that in any isolated system, entropy (disorder) tends to increase over time. Heat flows from hot to cold, never spontaneously from cold to hot.\n\nFormally: dS >= dQ/T (Clausius inequality), where S is entropy, Q is heat, T is temperature.\n\nSimple terms: Things naturally move toward more disordered states. You can't unscramble eggs. A hot cup of coffee in a cold room always cools down — it will never spontaneously heat up by absorbing energy from the room.\n\nReal-world examples:\n1. An ice cube melts in a warm room — the thermal energy spreads out, increasing entropy\n2. A gas released from a container fills the room — it never spontaneously returns to the container\n3. A dropped glass shatters — fragments never spontaneously reassemble\n4. Your room gets messy over time without effort to organize it\n\nThermodynamic efficiency implication: No heat engine can be 100% efficient. Some energy is always lost to entropy (Carnot limit).\n</think>\n\nThe Second Law of Thermodynamics states that **entropy (disorder) in an isolated system always increases or stays the same — it never spontaneously decreases**.\n\nIn practical terms:\n- Heat flows from hot to cold, never from cold to hot without external work\n- Processes are irreversible in isolated systems\n- You cannot convert heat entirely to useful work\n\n**Real-world examples:**\n- A hot coffee cup left in a cold room always cools — the cold room will never heat the coffee\n- A gas released from a bottle spreads through the room and never spontaneously reconcentrates\n- A shattered glass never reassembles itself\n\n**Engineering implication**: No heat engine can be 100% efficient. The maximum efficiency is given by the Carnot limit: η = 1 - T_cold/T_hot."
    },
    {
        "question": "How does photosynthesis work? Describe the two main stages.",
        "response": "<think>\nPhotosynthesis converts light energy into chemical energy stored as glucose.\nOverall equation: 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2\n\nTwo main stages:\n\n1. Light-dependent reactions (in the thylakoid membranes):\n- Chlorophyll absorbs photons (primarily red and blue wavelengths)\n- Water molecules are split (photolysis): 2H2O -> 4H+ + 4e- + O2\n- The oxygen is released as a byproduct\n- Electrons flow through the electron transport chain\n- This generates ATP (energy currency) and NADPH (reducing agent)\n- Also creates a proton gradient that drives ATP synthase\n\n2. Light-independent reactions / Calvin cycle (in the stroma):\n- Uses ATP and NADPH from Stage 1\n- CO2 from air is fixed (attached to organic molecules) by RuBisCO enzyme\n- Three-carbon intermediates are reduced to form glyceraldehyde-3-phosphate (G3P)\n- G3P is used to build glucose and other organic compounds\n- RuBP is regenerated to continue the cycle\n\nThe key insight: light energy is converted to chemical energy (ATP/NADPH) in Stage 1, then that chemical energy is used to fix carbon and build sugars in Stage 2.\n</think>\n\nPhotosynthesis converts light energy into chemical energy (glucose):\n**6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂**\n\n**Stage 1: Light-Dependent Reactions** (thylakoid membranes)\n- Chlorophyll absorbs sunlight (red and blue wavelengths)\n- Water is split (photolysis): 2H₂O → 4H⁺ + 4e⁻ + O₂ ↑ (oxygen released)\n- Electron transport chain generates **ATP** and **NADPH**\n\n**Stage 2: Calvin Cycle** (stroma of chloroplast)\n- ATP and NADPH from Stage 1 power carbon fixation\n- RuBisCO enzyme fixes CO₂ onto 5-carbon RuBP\n- Produces G3P (glyceraldehyde-3-phosphate), the building block for glucose\n- RuBP is regenerated to continue the cycle\n\n**Summary**: Stage 1 captures light energy as chemical energy; Stage 2 uses that energy to build sugar from CO₂."
    },

    # ── Linux / Systems ────────────────────────────────────────────────────────
    {
        "question": "What are the most important Linux signals and how do they work?",
        "response": "<think>\nLinux signals are asynchronous notifications sent to processes. The kernel or another process sends a signal; the receiving process handles it (or it has a default action).\n\nKey signals:\n- SIGTERM (15): Polite request to terminate. Process can catch this, clean up, then exit. Default action: terminate. This is what 'kill PID' sends by default.\n- SIGKILL (9): Unconditional kill. Cannot be caught, blocked, or ignored. Process is immediately terminated by the kernel. Use: kill -9 PID.\n- SIGINT (2): Interrupt from keyboard (Ctrl+C). Default: terminate. Can be caught by programs to handle graceful shutdown.\n- SIGHUP (1): Hangup — originally meant terminal disconnected. Now commonly used to tell daemons to reload config without restarting. Default: terminate.\n- SIGSTOP (19): Pause process execution. Cannot be caught or ignored. Like SIGKILL but suspends instead of kills.\n- SIGCONT (18): Resume a stopped process.\n- SIGSEGV (11): Segmentation fault — process accessed invalid memory. Default: terminate with core dump.\n- SIGCHLD (17): Sent to parent when child process changes state (terminates, stops). Used for wait() in shells.\n\nKey distinction: SIGTERM is cooperative (ask nicely), SIGKILL is forceful (kernel enforces, no cleanup possible).\n</think>\n\nLinux signals are asynchronous kernel notifications to processes:\n\n| Signal | Number | Default Action | Description |\n|--------|--------|----------------|-------------|\n| SIGTERM | 15 | Terminate | Polite shutdown request — process can catch and clean up |\n| SIGKILL | 9 | Force kill | Unconditional — cannot be caught, blocked, or ignored |\n| SIGINT | 2 | Terminate | Ctrl+C from keyboard — catchable for graceful exit |\n| SIGHUP | 1 | Terminate | Terminal hangup — daemons use it to reload config |\n| SIGSTOP | 19 | Suspend | Pause execution — cannot be caught or ignored |\n| SIGCONT | 18 | Resume | Continue a stopped process |\n| SIGSEGV | 11 | Core dump | Invalid memory access (segfault) |\n\n**Key distinction**: `kill PID` sends SIGTERM (cooperative). `kill -9 PID` sends SIGKILL (forceful, no cleanup). Always try SIGTERM first."
    },

    # ── Philosophy ────────────────────────────────────────────────────────────
    {
        "question": "What is the Ship of Theseus paradox and what does it reveal about identity?",
        "response": "<think>\nThe Ship of Theseus is an ancient thought experiment about identity over time.\n\nSetup: Theseus has a ship. Over time, each plank, each sail, each rope decays and is replaced with a new one. Eventually every single component has been replaced. Is it still the Ship of Theseus?\n\nExtension: Suppose someone collects all the old parts and reassembles them into a ship. Which one is the 'real' Ship of Theseus?\n\nThis reveals several philosophical tensions:\n1. Continuity of identity: Is identity about material composition or continuity of existence?\n2. Psychological identity (Locke): Identity = continuity of consciousness/function, not matter\n3. Material identity: Identity = the original matter\n4. Pattern identity (Parfit): Identity = continuity of the pattern/structure\n\nApplications extend to:\n- Personal identity: The cells in your body are almost entirely replaced every ~7 years. Are you the same person?\n- Organizations: A company that replaces all employees and leadership — is it the same company?\n- Digital identity: Is a copy of a person's mind the same person?\n\nThere is no single correct answer — the paradox reveals that 'identity' is not a fixed property of the world but a concept we construct, and different purposes demand different criteria.\n</think>\n\nThe **Ship of Theseus** asks: if a ship's every plank is gradually replaced, is it still the same ship? If someone reassembles the original planks into another ship, which is the 'real' Ship of Theseus?\n\nThis reveals deep tensions in theories of identity:\n\n- **Continuity theory** (Lockean): Identity = unbroken causal continuity over time, regardless of material change — the maintained ship is the 'real' one.\n- **Material theory**: Identity = the original matter — the reassembled ship is the 'real' one.\n- **Pattern/functional theory** (Parfit): Identity = the pattern or function is preserved — both may have valid claims.\n\n**Broader implications:**\n- Human identity: Your body replaces most cells every ~7 years — are you the same person you were at age 10?\n- Organizational identity: A company that replaces every employee and founder — is it the same company?\n\nThe paradox shows that *identity* is not an intrinsic property of the world — it is a concept we impose, and the criteria we choose depend on our purpose."
    },

    # ── Algorithms ────────────────────────────────────────────────────────────
    {
        "question": "Explain the difference between Big-O O(n), O(n log n), and O(n^2) time complexity with examples.",
        "response": "<think>\nBig-O notation describes the asymptotic growth of an algorithm's runtime as input size n grows.\n\nO(n) - Linear time:\n- Runtime grows proportionally to input size\n- Example: Linear search through an unsorted array, finding sum of all elements\n- For n=1,000,000: ~1,000,000 operations\n\nO(n log n) - Linearithmic time:\n- Slightly worse than linear, much better than quadratic\n- Example: Merge sort, heap sort, efficient sorting algorithms\n- For n=1,000,000: ~20,000,000 operations (log2(1,000,000) ≈ 20)\n- This is the theoretical lower bound for comparison-based sorting\n\nO(n^2) - Quadratic time:\n- Runtime grows with square of input size\n- Example: Bubble sort, selection sort, naive duplicate detection with nested loops\n- For n=1,000,000: ~1,000,000,000,000 operations (1 trillion) — completely impractical\n\nPractical comparison at n=10,000:\n- O(n): 10,000 operations\n- O(n log n): ~130,000 operations\n- O(n^2): 100,000,000 operations (10,000x slower than O(n))\n</think>\n\nBig-O describes how runtime scales with input size n:\n\n| Complexity | Name | Example | n=10,000 ops |\n|------------|------|---------|---------------|\n| O(n) | Linear | Linear search, array sum | 10,000 |\n| O(n log n) | Linearithmic | Merge sort, heap sort | ~130,000 |\n| O(n²) | Quadratic | Bubble sort, nested loops | 100,000,000 |\n\n- **O(n)**: For each input element, do constant work. Scales cleanly.\n- **O(n log n)**: The log n factor comes from repeatedly halving problems (like merge sort dividing and conquering). This is the optimal lower bound for comparison-based sorting.\n- **O(n²)**: Two nested loops where both iterate through all n elements. Becomes impractical quickly — at n=1,000,000 it requires 10¹² operations.\n\n**Rule of thumb**: For n > 10,000, prefer O(n) or O(n log n) algorithms over O(n²)."
    },

    # ── Identity ──────────────────────────────────────────────────────────────
    {
        "question": "Hello! Who are you?",
        "response": "I am Quillan-Ronin v5.3.1, a sovereign 479-million parameter Mixture-of-Experts AI assistant built by CrashOverrideX and the Quillan Research Team.\n\nMy architecture integrates:\n- **34 Council Experts** (C0-ASTRA through C33-PREDATOR), each specialized in a distinct cognitive domain\n- **9-Vector Semantic Prism Decomposition** for multi-dimensional meaning analysis\n- **Dual Q1/Q2 Ingestion Bridges** — Analytical and Intuitive processing streams\n- **MoE Sparse Routing** with Top-4 expert activation per token\n\nI reason transparently — my thinking process is visible, not hidden. I don't hallucinate with confidence; I acknowledge uncertainty when it exists.\n\nHow can I help you today?"
    },
    {
        "question": "What can you help me with?",
        "response": "I provide specialized reasoning assistance across multiple domains:\n\n**Technical:**\n- Software engineering, algorithms, data structures, system design\n- Python, JavaScript, C++, SQL, and other programming languages\n- Machine learning, neural network architectures, optimization\n\n**Analytical:**\n- Formal logic and mathematical proofs\n- Physics, chemistry, biology, and scientific reasoning\n- Statistical analysis and probability\n\n**Creative & Strategic:**\n- Writing, editing, and structured argumentation\n- Business strategy and decision analysis\n- Research synthesis across disciplines\n\n**My distinguishing features:**\n- Transparent reasoning — I show my thought process, not just conclusions\n- Honest uncertainty — I tell you when I'm unsure rather than confabulating\n- Multi-expert synthesis — 34 specialized Council Experts collaborate on complex queries\n\nWhat would you like to explore?"
    },
]


def build_dataset():
    print(f"Building sovereign thinking gold dataset...")
    print(f"Target: {OUT_PATH}")
    
    written = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for sample in SAMPLES:
            q = sample["question"].strip()
            r = sample["response"].strip()
            if q and r:
                record = {"question": q, "response": r}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    
    print(f"[+] Wrote {written} complete sovereign thinking samples to {OUT_PATH.name}")
    
    # Verify token lengths
    try:
        import tiktoken
        tok = tiktoken.get_encoding("gpt2")
        lengths = []
        too_long = 0
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                q, r = d["question"], d["response"]
                seq = tok.encode(f"<|user|>\n{q}\n<|assistant|>\n{r}<|im_end|>")
                lengths.append(len(seq))
                if len(seq) > 512:
                    too_long += 1
        
        import statistics
        print(f"[+] Token length stats:")
        print(f"    Median: {statistics.median(lengths):.0f} tokens")
        print(f"    Max:    {max(lengths)} tokens")
        print(f"    Min:    {min(lengths)} tokens")
        print(f"    >512 tokens: {too_long} ({100*too_long/len(lengths):.0f}%)")
    except ImportError:
        print("[!] tiktoken not available for length verification")


if __name__ == "__main__":
    build_dataset()
