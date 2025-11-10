from transformers import pipeline
import torch

# Using a smaller 1B model that fits in memory
model_id = "ibm-granite/granite-4.0-h-1b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    dtype=torch.bfloat16,  # More memory efficient
    device_map="auto",  # Re-enabled for efficient memory management
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)

print(outputs[0]["generated_text"][-1])

# {
#   'role': 'assistant',
#   'content': 'Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at the scale of atoms and subatomic particles. It introduces concepts like wave-particle duality, quantization, superposition, and entanglement. Key principles include the uncertainty principle (position and momentum cannot both be precisely defined), the probabilistic nature of particles, and the conservation of energy.'
# }
