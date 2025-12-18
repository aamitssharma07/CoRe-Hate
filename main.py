# main.py
import pandas as pd

# Import from our modules
import config
import utils
import prompts

from langchain_core.output_parsers import StrOutputParser


def run_experiment_for_model(model_name):
    all_data_rows = []

    # 1. Load Model
    try:
        model, tokenizer = utils.load_model_and_tokenizer(model_name)
    except Exception as e:
        print(f"[ERROR] Could not load {model_name}: {e}")
        return

    # 2. Create Pipelines (must return a LangChain Runnable/LLM)
    llm_creative = utils.create_pipeline(model, tokenizer, temperature=0.9)
    llm_strict = utils.create_pipeline(model, tokenizer, temperature=0.1)

    # 3. LCEL Chains (v1+): add StrOutputParser so outputs are always strings
    chain_transfer = prompts.scenario_prompt | llm_creative | StrOutputParser()
    chain_counterfactual = prompts.counterfactual_prompt | llm_strict | StrOutputParser()

    print(f"[EXPERIMENT] Generating data with {model_name}...")

    # 4. Main Loop
    for seed in config.SEED_POSTS:
        seed_id = str(seed["id"])
        print(f"Processing Seed ID {seed_id}...")

        # Save Original Seed
        all_data_rows.append(
            {
                "id": seed_id,
                "parent_id": "root",
                "text": seed["text"],
                "label": seed.get("label", "seed"),
                "target_group": seed.get("target", ""),
                "implied_statement": seed.get("implied_statement", ""),
                "category": "seed_original",
                "model": "human_gold",
            }
        )

        try:
            # Step A: Scenario Transfer (invoke -> str because of StrOutputParser)
            response_text = chain_transfer.invoke(
                {
                    "seed_text": seed["text"],
                    "target_group": seed.get("target", ""),
                    "implied_statement": seed.get("implied_statement", ""),
                }
            )

            generated_posts = [
                line.strip()
                for line in str(response_text).split("\n")
                if len(line.strip()) > 15
            ][:5]

            for idx, gen_post in enumerate(generated_posts, start=1):
                gen_id = f"{seed_id}.{idx}"

                all_data_rows.append(
                    {
                        "id": gen_id,
                        "parent_id": seed_id,
                        "text": gen_post,
                        "label": "generated",
                        "target_group": seed.get("target", ""),
                        "implied_statement": seed.get("implied_statement", ""),
                        "category": "generated",
                        "model": model_name,
                    }
                )

                # Step B: Counterfactual / Rewrite
                # FIXED: Changed key from "source_text" to "hate_post" to match prompts.py
                rewritten_text = chain_counterfactual.invoke({"hate_post": gen_post}).strip()

                cf_id = f"{gen_id}_cf"
                all_data_rows.append(
                    {
                        "id": cf_id,
                        "parent_id": gen_id,
                        "text": rewritten_text,
                        "label": "rewrite",
                        "target_group": seed.get("target", ""),
                        "implied_statement": "N/A",
                        "category": "rewrite",
                        "model": model_name,
                    }
                )

        except Exception as e:
            print(f"   Error on seed {seed_id}: {e}")

    # 5. Save and Cleanup
    clean_name = model_name.replace("/", "_")
    filename = f"results_{clean_name}.csv"

    df = pd.DataFrame(all_data_rows).sort_values(by="id")
    df.to_csv(filename, index=False)
    print(f"[SUCCESS] Saved {len(df)} rows to {filename}")

    del model, tokenizer, llm_creative, llm_strict, chain_transfer, chain_counterfactual
    utils.clean_memory()


if __name__ == "__main__":
    utils.setup_reproducibility(seed=42)
    for model in config.MODELS_TO_TEST:
        run_experiment_for_model(model)