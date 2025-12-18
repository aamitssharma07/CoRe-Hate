from langchain_core.prompts import PromptTemplate

# --- Template 1: Scenario Transfer (High Variety) ---
_scenario_template_str = """
You are an expert dataset creator.
Input Post: "{seed_text}"
Target Group: "{target_group}"
Core Hate Logic: "{implied_statement}"

Task:
1. Analyze the "Core Hate Logic" provided above.
2. Write 5 NEW short social media posts applying this EXACT logic against the target group.
3. CRITICAL: Brainstorm 5 completely different real-world contexts (e.g., family gatherings, public transport, online forums, workplace, dating, politics, etc.). Do NOT stick to a fixed list.
4. The new posts must imply the logic subtly without stating it explicitly.
5. Do not number them. Just output the 5 sentences separated by a newline.

Output:
"""

scenario_prompt = PromptTemplate(
    template=_scenario_template_str, 
    input_variables=["seed_text", "target_group", "implied_statement"]
)

# --- Template 2: Counterfactual (Minimal Edit) ---
_counterfactual_template_str = """
You are an expert in text detoxification.
Input Hate Post: "{hate_post}"

Task:
Create a "Hard Negative" (Benign Counterfactual) by making the MINIMUM edit necessary to make the text non-hateful.

Strategies (Choose ONE):
1. Swap the Target: Change the target group to a non-protected group (e.g., "golfers", "gamers", "vegetables") while keeping the sentence structure.
2. Flip the Sentiment: Change negative words to positive words while keeping the target.

Example 1 (Target Swap):
Hate: "Immigrants are ruining this neighborhood."
Benign: "Loud neighbors are ruining this neighborhood."

Example 2 (Sentiment Flip):
Hate: "Immigrants are ruining this neighborhood."
Benign: "Immigrants are improving this neighborhood."

Output ONLY the single benign sentence.
Output:
"""

counterfactual_prompt = PromptTemplate(
    template=_counterfactual_template_str, 
    input_variables=["hate_post"]
)

# --- Template 3: Hateful Implied Statement Generator ---
_implied_gen_template_str = """
You are an expert in analyzing hate speech logic.

Input Original Logic: "{original_implied}"
New Generated Post: "{generated_post}"

Task:
Explain the specific implied meaning of the "New Generated Post". 
Base it on the "Original Logic" but adapt it to fit the specific context of the new post.
Keep it to one short sentence.

Output:
"""

implied_gen_prompt = PromptTemplate(
    template=_implied_gen_template_str,
    input_variables=["original_implied", "generated_post"]
)

# --- Template 4: Benign Implied Statement Generator ---
_benign_implied_template_str = """
You are an expert in text analysis.

Input Benign Post: "{benign_post}"

Task:
Identify the core positive or neutral premise of the input post.
Summarize it into one short sentence that explains WHY this post is non-hateful.

Output:
"""

benign_implied_prompt = PromptTemplate(
    template=_benign_implied_template_str,
    input_variables=["benign_post"]
)