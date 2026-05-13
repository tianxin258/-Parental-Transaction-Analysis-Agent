import os
from openai import OpenAI

client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
resp = client.chat.completions.create(
    model=os.environ["LLM_MODEL"],
    messages=[{"role": "user", "content": 'Say hello in JSON: {"msg": "hi"}'}],
    max_tokens=100
)
print("LLM OK:", resp.choices[0].message.content[:150])
