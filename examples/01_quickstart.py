"""Quickstart: Multilingual chat completion with N-ATLAS."""

from natlas import Natlas

# Connect to local or remote N-ATLAS instance
client = Natlas(base_url="http://127.0.0.1:8000/v1")

# 1. Chat in Yoruba with automatic tone & diacritic enforcement
print("--- Yoruba Chat ---")
response_yo = client.chat.create(
    language="yoruba",
    tone_enforcement=True,
    messages=[{"role": "user", "content": "Bawo ni mo se le kọ iwe ni èdè Yorùbá?"}],
)
print(response_yo.choices[0].message.content)

# 2. Chat in Hausa
print("\n--- Hausa Chat ---")
response_ha = client.chat.create(
    language="hausa",
    tone_enforcement=True,
    messages=[{"role": "user", "content": "Barka dai! Yaya kake?"}],
)
print(response_ha.choices[0].message.content)

# 3. Chat in Igbo
print("\n--- Igbo Chat ---")
response_ig = client.chat.create(
    language="igbo",
    tone_enforcement=True,
    messages=[{"role": "user", "content": "Kedu ka ị mere taa?"}],
)
print(response_ig.choices[0].message.content)

client.close()
