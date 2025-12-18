import pandas as pd
import os

# Import from our modules
import config
import utils
import prompts

from langchain_core.output_parsers import StrOutputParser

def run_experiment_for_model(model_name):
    all_data_rows = []

    # --- 0. Setup Output Folder ---
    output_dir = "Results"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Model
    try:
        model, tokenizer = utils.load_model_and_tokenizer(model_name)
    except Exception as e:
        print(f"[ERROR] Could not load {model_name}: {e}")
        return

    # 2. Create Pipelines
    # Creative for generating new scenarios
    llm_creative = utils.create_pipeline(model, tokenizer, temperature=0.9)
    # Strict for counterfactuals and logic extraction
    llm_strict = utils.create_pipeline(model, tokenizer, temperature=0.1)

    # 3. Define Chains (Sequential Approach)
    # A. Scenario Transfer
    chain_transfer = prompts.scenario_prompt | llm_creative | StrOutputParser()
    
    # B. Counterfactual Generation
    chain_counterfactual = prompts.counterfactual_prompt | llm_strict | StrOutputParser()
    
    # C. Hateful Logic Extraction (For Variations)
    chain_hate_implied = prompts.implied_gen_prompt | llm_strict | StrOutputParser()

    # D. Benign Logic Extraction (For Counterfactuals)
    chain_benign_implied = prompts.benign_implied_prompt | llm_strict | StrOutputParser()

    print(f"[EXPERIMENT] Generating data with {model_name}...")

    # 4. Main Loop
    for seed in config.SEED_POSTS:
        seed_id = str(seed["id"])
        print(f"Processing Seed ID {seed_id}...")

        # --- PROCESS STEP 1: The Original Seed ---
        
        # 1a. Store Original Seed
        all_data_rows.append({
            "id": seed_id,
            "parent_id": "root",
            "text": seed["text"],
            "label": seed.get("label", "implicit_hate"),
            "target_group": seed.get("target", ""),
            "implied_statement": seed.get("implied_statement", ""),
            "category": "seed_original",
            "model": "human_gold",
        })

        # 1b. Generate Counterfactual for ORIGINAL Seed
        try:
            # Generate the Counterfactual
            cf_seed_text = chain_counterfactual.invoke({"hate_post": seed["text"]}).strip()
            
            # Generate Implied Meaning for this specific Benign Post
            cf_seed_implied = chain_benign_implied.invoke({"benign_post": cf_seed_text}).strip()

            all_data_rows.append({
                "id": f"{seed_id}_cf",
                "parent_id": seed_id,
                "text": cf_seed_text,
                "label": "not_hate",
                "target_group": seed.get("target", ""),
                "implied_statement": cf_seed_implied, 
                "category": "counterfactual_of_seed",
                "model": model_name,
            })
        except Exception as e:
            print(f"   Error generating seed CF: {e}")

        # --- PROCESS STEP 2: Generate Variations ---
        try:
            # Generate 5 new hate posts
            response_text = chain_transfer.invoke({
                "seed_text": seed["text"],
                "target_group": seed.get("target", ""),
                "implied_statement": seed.get("implied_statement", ""),
            })

            generated_posts = [
                line.strip()
                for line in str(response_text).split("\n")
                if len(line.strip()) > 15
            ][:5]

            for idx, gen_post in enumerate(generated_posts, start=1):
                gen_id = f"{seed_id}.{idx}"
                
                # 2a. Generate Implied Statement for the Generated Hate Post
                # We ask the LLM: "Given the original logic, what does THIS specific post mean?"
                try:
                    new_hate_implied = chain_hate_implied.invoke({
                        "original_implied": seed.get("implied_statement", ""),
                        "generated_post": gen_post
                    }).strip()
                except:
                    new_hate_implied = seed.get("implied_statement", "") # Fallback

                # Store Generated Hate Post
                all_data_rows.append({
                    "id": gen_id,
                    "parent_id": seed_id,
                    "text": gen_post,
                    "label": "implicit_hate",
                    "target_group": seed.get("target", ""),
                    "implied_statement": new_hate_implied,
                    "category": "generated_hate",
                    "model": model_name,
                })

                # 2b. Generate Counterfactual for this New Post
                rewritten_text = chain_counterfactual.invoke({"hate_post": gen_post}).strip()

                # 2c. Generate Implied Meaning for the Benign Counterfactual
                try:
                    new_benign_implied = chain_benign_implied.invoke({"benign_post": rewritten_text}).strip()
                except:
                    new_benign_implied = "Benign sentiment rewrite."

                cf_id = f"{gen_id}_cf"
                all_data_rows.append({
                    "id": cf_id,
                    "parent_id": gen_id,
                    "text": rewritten_text,
                    "label": "not_hate",
                    "target_group": seed.get("target", ""),
                    "implied_statement": new_benign_implied,
                    "category": "counterfactual_of_generated",
                    "model": model_name,
                })

        except Exception as e:
            print(f"   Error on seed {seed_id} loop: {e}")

    # 5. Save and Cleanup
    clean_name = model_name.replace("/", "_")
    filename = f"{output_dir}/results_{clean_name}.csv"

    df = pd.DataFrame(all_data_rows).sort_values(by="id")
    df.to_csv(filename, index=False)
    print(f"[SUCCESS] Saved {len(df)} rows to {filename}")

    # Explicit cleanup to free VRAM for next model
    del model, tokenizer, llm_creative, llm_strict 
    del chain_transfer, chain_counterfactual, chain_hate_implied, chain_benign_implied
    utils.clean_memory()


if __name__ == "__main__":
    utils.setup_reproducibility(seed=42)
    for model in config.MODELS_TO_TEST:
        run_experiment_for_model(model)