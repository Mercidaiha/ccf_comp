import pandas as pd
import random

model_names = [
    "qwen25_72b_instruct",
    "gpt_4o_mini_cot",
    "ministral_8b_instruct_2410",
    "deepseek_chat",
    "glm_4_plus",
    "llama31_8b_instruct",
    "qwen25_32b_int4",
    "gpt_4o",
    "glm_4_air",
    "gpt_4o_mini",
    "qwen25_math_7b_instruct",
    "llama31_70b_instruct",
    "mistral_7b_instruct_v02",
    "mixtral_8x7b_instruct",
    "glm_4_flash",
    "qwq_32b_preview",
    "gemini15_flash",
    "deepseek_coder",
    "qwen25_7b_instruct",
    "llama31_405b_instruct"
]

if __name__ == "__main__":
    task = [
        'aclue',
        'arc_c',
        'cmmlu',
        'hotpot_qa',
        'math',
        'mmlu',
        'squad'
    ]

    for t in task:
        data_df = f"competition_data/raw_data/{t}_train.csv"
        data_df = pd.read_csv(data_df)
        data_df = data_df[model_names]
        means = data_df[model_names].mean()
        best_model = means.idxmax()
        print(f"Best model for {t}: {best_model} with mean score {means[best_model]:.3f}")

