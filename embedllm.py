import argparse
import random
import torch
import pandas as pd
import numpy as np
import time
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import Adam
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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

historical_best_in_train = {
    'aclue': 'glm_4_plus',
    'arc_c': "glm_4_plus",
    'cmmlu': 'qwen25_72b_instruct',
    'hotpot_qa': 'gpt_4o',
    'math': 'deepseek_coder',
    'mmlu': 'llama31_405b_instruct',
    'squad': 'llama31_405b_instruct'
}

class TextMF(nn.Module):
    def __init__(self, question_embeddings, model_embedding_dim, alpha, num_models, num_prompts, text_dim=384, num_classes=2):
        super(TextMF, self).__init__()
        # Model embedding network
        self.P = nn.Embedding(num_models, model_embedding_dim)

        # Question embedding network
        self.Q = nn.Embedding(num_prompts, text_dim).requires_grad_(False)
        self.Q.weight.data.copy_(question_embeddings)
        self.text_proj = nn.Linear(text_dim, model_embedding_dim)

        # Noise/Regularization level
        self.alpha = alpha
        self.classifier = nn.Linear(model_embedding_dim, num_classes)

    def forward(self, model, prompt, test_mode=False):
        p = self.P(model)
        q = self.Q(prompt)
        if not test_mode:
            # Adding a small amount of noise in question embedding to reduce overfitting
            q += torch.randn_like(q) * self.alpha
        q = self.text_proj(q)
        return self.classifier(p * q)
    
    @torch.no_grad()
    def predict(self, model, prompt):
        logits = self.forward(model, prompt, test_mode=True) # During inference no noise is applied
        return torch.argmax(logits, dim=1)

class KNN(nn.Module):
    def __init__(self, query_embeddings, best_models, num_neighbors=5):
        super(KNN, self).__init__()
        self.train_x = query_embeddings
        self.train_y = [','.join(models) for models in best_models]
        self.num_neighbors = num_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=num_neighbors)
        self.knn.fit(self.train_x, self.train_y)

    def forward(self, test_query_embedding):
        _, indices = self.knn.kneighbors(test_query_embedding)
        neighbor_labels = []
        idx_list = indices[0]
        neighbor_labels.extend([self.train_y[i] for i in idx_list])
        model_counts = {}
        for label in neighbor_labels:
            models = label.split(',')
            for model in models:
                model_counts[model] = model_counts.get(model, 0) + 1
        return max(model_counts.items(), key=lambda x: x[1])[0]


class CustomDataset(Dataset):
    def __init__(self, models, prompts, labels, type='train'):
        self.models = models
        self.prompts = prompts
        self.labels = labels
        self.type = type
    
    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):
        return self.models[index], self.prompts[index], self.labels[index]

    def get_dataloaders(self, batch_size):
        return DataLoader(self, batch_size, shuffle=False)

def load_and_process_data(train_data, model_names, batch_size=64):
    train_len = int(len(train_data) * 0.9)
    train_models = []
    train_prompts = []
    train_labels = []
    for i, model_name in enumerate(model_names):
        for question_id, label in zip(train_data[:train_len]['id'], train_data[:train_len][model_name]):
            train_models.append(i)
            train_prompts.append(question_id)
            train_labels.append(label)
    train_dataset = CustomDataset(train_models, train_prompts, train_labels, 'train')

    eval_models = []
    eval_prompts = []
    eval_labels = []
    for i, model_name in enumerate(model_names):
        for question_id, label in zip(train_data[train_len:]['id'], train_data[train_len:][model_name]):
            eval_models.append(i)
            eval_prompts.append(question_id)
            eval_labels.append(label)
    eval_dataset = CustomDataset(eval_models, eval_prompts, eval_labels, 'eval')
    train_loader = train_dataset.get_dataloaders(batch_size)
    eval_loader = eval_dataset.get_dataloaders(batch_size)
    return train_loader, eval_loader
    
def evaluate(net, test_loader, device, model_num=20):
    """Unified evaluation function that routes to specific evaluator based on mode"""
    return evaluator_router(net, test_loader, [device], model_num)

def evaluator_router(net, test_iter, devices, model_num=20):
    net.eval()
    scores = []
    correctness_result = {}
    with torch.no_grad():
        for i, (prompts, models, labels, categories) in enumerate(test_iter):
            prompts = prompts.to(devices[0])
            models = models.to(devices[0])
            labels = labels.to(devices[0])
            categories = categories.to(devices[0])
            logits = net(models, prompts)
            logit_diff = (logits[:, 1] - logits[:, 0])
            max_index = torch.argmax(logit_diff)
            scores.append(labels[max_index].item())
            correctness_result[int(prompts[0])] = int(labels[max_index] == 1)
    net.train()
    return np.mean(scores)

def create_router_dataloader(original_dataloader):
    # Concatenate all batches
    all_models, all_prompts, all_labels = [], [], []
    for models, prompts, labels in original_dataloader:
        all_models.append(models)
        all_prompts.append(prompts)
        all_labels.append(labels)
    
    all_models = torch.cat(all_models)
    all_prompts = torch.cat(all_prompts)
    all_labels = torch.cat(all_labels)

    # Create label dictionary
    label_dict = {}
    for i in range(len(all_prompts)):
        prompt_id = int(all_prompts[i])
        model_id = int(all_models[i])
        label = int(all_labels[i])
        
        if prompt_id not in label_dict:
            label_dict[prompt_id] = {}
        label_dict[prompt_id][model_id] = label

    # Get unique prompts and models
    unique_prompts = sorted(set(all_prompts.tolist()))
    unique_models = sorted(set(all_models.tolist()))
    model_num = len(unique_models)

    # Build router dataloader content
    new_models, new_prompts, new_labels = [], [], []
    for prompt_id in unique_prompts:
        prompt_tensor = torch.tensor([prompt_id] * model_num)
        model_tensor = torch.tensor(unique_models)
        label_tensor = torch.tensor([label_dict[prompt_id].get(model_id, 0) for model_id in unique_models])
        
        new_prompts.append(prompt_tensor)
        new_models.append(model_tensor)
        new_labels.append(label_tensor)

    # Concatenate tensors
    new_prompts = torch.cat(new_prompts)
    new_models = torch.cat(new_models)
    new_labels = torch.cat(new_labels)
    
    # Add dummy categories (all zeros) since they're not used in current implementation
    new_categories = torch.zeros_like(new_labels)

    # Create router dataloader
    router_dataset = TensorDataset(new_prompts, new_models, new_labels, new_categories)
    router_dataloader = DataLoader(router_dataset, batch_size=model_num, shuffle=False)
    return router_dataloader

# Main training loop
def train(net, train_loader, test_loader, num_epochs, lr, device, model_num=20, weight_decay=1e-5):
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    progress_bar = tqdm(total=num_epochs)

    best_res = 0
    best_model = None
    for _ in range(num_epochs):
        net.train()
        total_loss = 0
        for models, prompts, labels in train_loader:
            models, prompts, labels = models.to(device), prompts.to(device), (labels>0.5).long().to(device)
            optimizer.zero_grad()
            logits = net(models, prompts)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        route_acc = evaluate(net, test_loader, device, model_num)
        if route_acc > best_res:
            best_res = max(best_res, route_acc)
            best_model = net.state_dict().copy()
        progress_bar.set_postfix(train_loss=train_loss, route_acc=route_acc)
        progress_bar.update(1)
    return best_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--model-num", type=int, default=20)
    args = parser.parse_args()

    print("Loading dataset...")
    res = {}
    for task in [
        'aclue',
        'arc_c',
        'cmmlu',
        'hotpot_qa',
        'math',
        'mmlu',
        'squad'
    ]:
        print("-" * 100)
        print(f"Dealing with {task}...")
        train_data = pd.read_csv(f"competition_data/raw_data/{task}_train.csv")
        question_embeddings = f"competition_data/raw_data/question_embeddings_{task}.pth"
        question_embeddings = torch.load(question_embeddings)
        print(f"question_embeddings.shape: {question_embeddings.shape}")
        num_prompts = question_embeddings.shape[0]
        num_models = 20
        train_loader, eval_loader = load_and_process_data(train_data, model_names, batch_size=args.batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Initializing model...")
        model = TextMF(question_embeddings=question_embeddings, 
                    model_embedding_dim=args.embedding_dim, alpha=args.alpha,
                    num_models=num_models, num_prompts=num_prompts)
        model.to(device)

        print("Training model...")
        eval_loader = create_router_dataloader(eval_loader)
        best_model = train(model, train_loader, eval_loader, 
            num_epochs=args.num_epochs, 
            lr=args.learning_rate,
            device=device, 
            model_num=args.model_num
        )
        
        model.load_state_dict(best_model)
        best_models = []
        for i in range(len(train_data)):
            best_models.append(
                [model for model in model_names if train_data.iloc[i][model] == 1]
            )
        knn = KNN(query_embeddings=question_embeddings[:len(train_data)], best_models=best_models, num_neighbors=10)
        performance = []
        test_data = pd.read_csv(f"competition_data/raw_data/{task}_test_pred.csv")
        pred_models = []
        for i in range(len(test_data)):
            best_model = knn.forward(
                question_embeddings[len(train_data)+i].reshape(1, -1)
            )
            best_model = best_model if best_model in model_names else 'deepseek_chat'
            
            llm = torch.arange(len(model_names)).to(device)
            prompt = torch.tensor([len(train_data)+i]*len(model_names)).to(device)
            logits = model(llm, prompt, test_mode=True)
            logit_diff = torch.sigmoid(logits[:, 1] - logits[:, 0])
            max_index = torch.argmax(logit_diff)
            best_pred_model = model_names[max_index]
            best_logit_diff = logit_diff[max_index]
            
            final_selected_model = None
            if best_pred_model == best_model and historical_best_in_train[task] == best_model:
                final_selected_model = best_pred_model
            else:
                if best_logit_diff > 0.5:
                    final_selected_model = best_pred_model
                else:
                    models = [best_model, best_pred_model, historical_best_in_train[task]]
                    model_counts = {}
                    for m in models:
                        model_counts[m] = model_counts.get(m, 0) + 1
                    majority_model = max(model_counts.items(), key=lambda x: x[1])[0]
                    final_selected_model = majority_model
            pred_models.append(final_selected_model)
        test_data['pred'] = pred_models
        test_data.to_csv(f'competition_data/submit_data/{task}_test_pred.csv', index=False)
