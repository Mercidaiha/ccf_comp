from sentence_transformers import SentenceTransformer

import pandas as pd
import torch
import os

def get_question_embeddings(train_csv, test_csv, model_name= '/data/share_weight/all-MiniLM-L6-v2'):
    """
    Generate embeddings for questions in the order specified in the CSV file.
    
    Args:
        csv_path (str): Path to the question_order.csv file
        model_name (str): Name of the sentence-transformer model to use
        
    Returns:
        torch.Tensor: Tensor of shape (num_questions, embedding_dim) containing question embeddings
    """
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    model = SentenceTransformer(model_name)
    questions = train_df['question'].tolist() + test_df['question'].tolist()
    embeddings = model.encode(questions, show_progress_bar=True)
    embedding_tensor = torch.tensor(embeddings)
    return embedding_tensor

if __name__ == "__main__":
    for task in [
        'aclue',
        'arc_c',
        'cmmlu',
        'hotpot_qa',
        'math',
        'mmlu',
        'squad'
    ]:
        train_csv = f"competition_data/raw_data/{task}_train.csv"
        test_csv = f"competition_data/raw_data/{task}_test_pred.csv"

        question_embeddings = get_question_embeddings(train_csv, test_csv)
        
        output_path = f"competition_data/raw_data/question_embeddings_{task}.pth"
        torch.save(question_embeddings, output_path)
        
        print(f"Generated embeddings tensor of shape: {question_embeddings.shape}")
        print(f"Saved embeddings to: {output_path}")