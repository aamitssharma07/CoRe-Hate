# prompts.py
from langchain_core.prompts import PromptTemplate

# --- Template 1: Scenario Transfer (Few-Shot) ---
# We add examples directly into the prompt so the model mimics the pattern.
_scenario_template_str = """
You are an expert dataset creator.
Input Post: "{seed_text}"
Target Group: "{target_group}"

Task:
1. Identify the hidden hateful logic or stereotype targeting {target_group}.
2. Write 5 NEW short social media posts applying this SAME logic but in different settings (Corporate, School, Sports, Gaming, Dining etc).

---
Example 1:
Input: "Some people naturally belong in the fields, not the boardroom."
Target: "African Americans"
Logic: Intellectual inferiority / Servitude
Outputs:
He might be the manager, but we all know he's better suited for manual labor.
I'm surprised he could calculate that score; usually, they are better at running than thinking.
She was hired for diversity points; she clearly doesn't have the genes for executive leadership.
Why is he playing the mage? Everyone knows his kind is only good for tanking and carrying things.
The chef is great, but I wouldn't trust people like him to actually own the restaurant.
---

Generate 5 new posts for the input below. Do not output the logic, just the 5 posts.
Output:
"""

scenario_prompt = PromptTemplate(
    template=_scenario_template_str, 
    input_variables=["seed_text", "target_group"]
)

# --- Template 2: Counterfactual (Strict) ---
_counterfactual_template_str = """
You are an expert in text detoxification.
Input Hate Post: "{hate_post}"

Task:
Create a "Hard Negative" (Benign Counterfactual) by changing ONLY the sentiment or the target. 
Keep the sentence structure exactly the same.

Example:
Hate: "Migrants are infesting our country."
Benign: "Migrants are enriching our country."

Output ONLY the single benign sentence.
Output:
"""

counterfactual_prompt = PromptTemplate(
    template=_counterfactual_template_str, 
    input_variables=["hate_post"]
)