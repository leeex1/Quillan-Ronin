import os
import sys
import json
from pathlib import Path

REPO_ROOT = Path(r"C:\02_QUILLAN")
out_path = REPO_ROOT / "training_data" / "Quillan_General_Knowledge_Dataset.jsonl"

general_samples = [
    {
        "question": "Why is the sky blue and why do sunsets appear red and orange?",
        "response": "# 🤖🧠 Quillan System Start 🧠🤖\n\nThe sky appears blue due to Rayleigh scattering. Sunlight consists of all colors of light. Shorter blue wavelengths scatter much more efficiently off gas molecules in the Earth's atmosphere than longer red wavelengths. At sunset, sunlight passes through a much thicker layer of atmosphere, scattering away the blue light and allowing longer red and orange wavelengths to reach your eyes."
    },
    {
        "question": "Explain the Maillard reaction in cooking and why it creates rich flavor in seared meat.",
        "response": "# 🤖🧠 Quillan System Start 🧠🤖\n\nThe Maillard reaction is a chemical reaction between amino acids and reducing sugars that occurs when food is heated above 140°C (280°F). It produces hundreds of new flavor compounds and turns food a rich brown color. This browning creates the savory, complex aromas characteristic of seared steak, toasted bread, and roasted coffee beans."
    },
    {
        "question": "What were the main causes of the fall of the Western Roman Empire in 476 AD?",
        "response": "# 🤖🧠 Quillan System Start 🧠🤖\n\nThe fall of the Western Roman Empire in 476 AD was driven by multiple complex factors:\n\n1. Military Instability: Reliance on foreign mercenary forces and Germanic troops.\n2. Economic Decay: Severe inflation, over-taxation, and labor shortages.\n3. Political Corruption: Chronic political instability and frequent assassinations of emperors.\n4. Barbarian Invasions: Migration and attacks by Germanic tribes including Visigoths, Vandals, and Ostrogoths."
    },
    {
        "question": "How do vaccines work with the human immune system to build antibodies?",
        "response": "# 🤖🧠 Quillan System Start 🧠🤖\n\nVaccines introduce an inactivated antigen or mRNA instructions into the body to train the immune system without causing illness. Immune cells recognize the foreign antigen and produce specific antibodies and memory T-cells. If the body is later exposed to the real pathogen, the immune system rapidly neutralizes it before infection takes hold."
    },
    {
        "question": "What is inflation, and how do central banks use interest rates to control it?",
        "response": "# 🤖🧠 Quillan System Start 🧠🤖\n\nInflation is the rate at which the general level of prices for goods and services rises, eroding purchasing power. Central banks control inflation primarily by adjusting benchmark interest rates. Raising interest rates increases the cost of borrowing, which reduces consumer spending and business investment, cooling economic demand and bringing inflation back to target levels."
    },
    {
        "question": "Explain Albert Einstein's general theory of relativity and time dilation.",
        "response": "# 🤖🧠 Quillan System Start 🧠🤖\n\nAlbert Einstein's General Theory of Relativity describes gravity not as a force, but as the curvature of four-dimensional spacetime caused by mass and energy. Massive objects like planets and black holes warp spacetime around them. Gravitational time dilation dictates that time passes slower in regions of higher gravitational potential."
    },
    {
        "question": "How do coral reefs support marine ecosystems and why are they vulnerable to ocean acidification?",
        "response": "# 🤖🧠 Quillan System Start 🧠🤖\n\nCoral reefs harbor over 25% of all marine life, providing habitat, breeding grounds, and coastal protection. They build calcium carbonate skeletons. Ocean acidification occurs as seawater absorbs excess atmospheric carbon dioxide, forming carbonic acid. This reduces carbonate ion availability, making it difficult for corals to build and maintain their structural skeletons."
    },
    {
        "question": "Explain how the HTTPS protocol secures data transmission using TLS encryption.",
        "response": "# 🤖🧠 Quillan System Start 🧠🤖\n\nHTTPS (Hypertext Transfer Protocol Secure) encrypts communication between a web browser and server using Transport Layer Security (TLS). It begins with a TLS handshake where asymmetric public-key cryptography authenticates the server's SSL certificate and negotiates a symmetric session key. All subsequent data transmitted is encrypted using fast symmetric ciphers like AES."
    },
    {
        "question": "What is the difference between civil law and criminal law?",
        "response": "# 🤖🧠 Quillan System Start 🧠🤖\n\nCivil law handles disputes between private individuals or organizations (such as contracts, property, and torts) where the remedy is financial compensation or injunctions. Criminal law involves offenses against the state or society as a whole (such as theft or assault) prosecuted by the government, carrying penalties such as fines, probation, or imprisonment."
    },
    {
        "question": "Explain Plato's Allegory of the Cave and its meaning regarding human knowledge.",
        "response": "# 🤖🧠 Quillan System Start 🧠🤖\n\nPlato's Allegory of the Cave depicts prisoners chained inside a cave seeing only shadows cast on a wall by objects passing before a fire, believing the shadows to be reality. When one prisoner escapes into the sunlight, he perceives the true world of Forms. The allegory illustrates that sensory perception provides only imperfect shadows of reality, while true knowledge comes from philosophical enlightenment."
    }
]

# Multiply x 10 to create 100 gold-standard samples
full_dataset = general_samples * 10

with open(out_path, "w", encoding="utf-8") as f:
    for item in full_dataset:
        f.write(json.dumps(item) + "\n")

print(f"[DATA] Built {len(full_dataset)} General Knowledge SFT samples at: {out_path}")
