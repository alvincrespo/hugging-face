import os
from huggingface_hub import InferenceClient

client = InferenceClient()

completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "How many 'G's in 'huggingface'?"
        }
    ],
)

print(completion.choices[0].message)


# Output
# ChatCompletionOutputMessage(
#   role='assistant',
#   content="There are **3** 'G' letters in “huggingface.”",
#   reasoning='We need to answer question: "How many \'G\'s in \'huggingface\'?" The word is \'huggingface\'. We count \'G\'s. \'huggingface\': letters: h u g g i n g f a c e. We see g appears in positions 3,4,7? Let\'s write: h(1) u(2) g(3) g(4) i(5) n(6) g(7) f(8) a(9) c(10) e(11). So \'g\' occurs 3 times. So answer: 3. Should we consider case? All lowercase. So answer is 3.',
#   tool_call_id=None,
#   tool_calls=None
# )

# References
# https://huggingface.co/docs/inference-providers/index?python-clients=huggingface_hub
# https://huggingface.co/docs/inference-providers/pricing
