import os
import math
import random
import sys
import subprocess
import importlib
import shutil
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from datasets import load_dataset
from rouge_score import rouge_scorer
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME_ENCODER = "bert-base-uncased"
MODEL_NAME_DECODER = "facebook/bart-base"
MAX_SENT_LEN = 128
MAX_DOC_SENT = 16
TOP_K = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-5
EPOCHS = 3
W_RANK = 1.0
W_GEN = 1.0
USE_PYG = True
USE_DDP = False
MIXED_PRECISION = True
ACCUM_STEPS = 8

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
if device.type == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# Install packages
packages = {
    'transformers': 'transformers',
    'datasets': 'datasets',
    'torch': 'torch',
    'tqdm': 'tqdm',
    'spacy': 'spacy',
    'sentencepiece': 'sentencepiece',
    'rouge-score': 'rouge_score',
    'scikit-learn': 'scikit-learn',
}
for pkg, mod in packages.items():
    try:
        importlib.import_module(mod)
    except ImportError:
        logger.info(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Install PyG or DGL
try:
    import torch_geometric
    logger.info("PyG already installed")
except ImportError:
    logger.info("Installing PyTorch Geometric...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric==2.3.0", "-f", "https://data.pyg.org/whl/torch-2.0.0+cu118.html"])
    except Exception as e:
        logger.error(f"PyG install failed: {e} — trying DGL...")
        try:
            import dgl
            logger.info("DGL already installed")
        except ImportError:
            logger.info("Installing DGL...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "dgl"])
            except Exception as e:
                logger.error(f"DGL install failed: {e}; install PyG or DGL manually.")

# Install spaCy model
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    logger.info(f"spaCy pipeline: {nlp.pipe_names}")
except Exception as e:
    logger.info("Installing spaCy model en_core_web_sm...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")
    logger.info(f"spaCy pipeline after install: {nlp.pipe_names}")

# Sentence splitting
def split_sentences(text, max_sents=MAX_DOC_SENT):
    if not isinstance(text, str) or not text.strip():
        logger.warning(f"Invalid input for sentence splitting: {text}")
        return []
    try:
        doc = nlp(text)
        sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        logger.debug(f"Split into {len(sents)} sentences")
        return sents[:max_sents] if len(sents) > max_sents else sents
    except Exception as e:
        logger.error(f"Sentence splitting failed for text: {text[:100]}... Error: {e}")
        return []

# Clear dataset cache
cache_dir = "C:\\Users\\PC\\.cache\\huggingface\\datasets"
logger.info(f"Clearing dataset cache at {cache_dir}...")
try:
    shutil.rmtree(cache_dir)
except FileNotFoundError:
    logger.info("Cache directory not found, proceeding with fresh download.")
os.makedirs(cache_dir, exist_ok=True)

# Load dataset
logger.info("Loading dataset...")
local_data_dir = "C:\\Users\\PC\\Desktop\\IRNLP PROJECT\\cnn_dailymail"
dataset = None
try:
    logger.info("Attempting to load CNN/DailyMail from local CSV files...")
    if os.path.exists(local_data_dir):
        data_files = {
            "train": os.path.join(local_data_dir, "train.csv"),
            "validation": os.path.join(local_data_dir, "validation.csv"),
            "test": os.path.join(local_data_dir, "test.csv"),
        }
        for split, file_path in data_files.items():
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing {split} file: {file_path}. Download from Kaggle and place in {local_data_dir}.")
        dataset = load_dataset("csv", data_files=data_files)
        # Filter out invalid articles
        dataset = dataset.filter(lambda x: x['article'] is not None and isinstance(x['article'], str) and x['article'].strip())
        logger.info(f"Loaded dataset from local CSV files: {local_data_dir}")
    else:
        raise FileNotFoundError(f"Local dataset directory {local_data_dir} not found. Download CSV files from Kaggle and place them there.")
except Exception as e:
    logger.error(f"Local CSV loading failed: {e}")
    logger.info("Falling back to small online subset for testing...")
    try:
        raw_train = load_dataset("cnn_dailymail", "1.0.0", split="train[:1000]", cache_dir=cache_dir, download_mode="force_redownload")
        raw_val = raw_train
        raw_test = raw_train
        from datasets import DatasetDict
        dataset = DatasetDict({"train": raw_train, "validation": raw_val, "test": raw_test})
        logger.info("Using CNN/DailyMail 1.0.0 subset (1000 samples) for testing.")
    except Exception as e2:
        raise RuntimeError(
            f"Failed to load any dataset: {e2}. "
            f"Please manually download the CSV files from https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail "
            f"and place train.csv, validation.csv, test.csv in {local_data_dir}, or ensure a stable internet connection for online loading."
        )

# Assign splits
raw_train = dataset['train']
raw_val = dataset['validation']
raw_test = dataset['test']
logger.info(f"Train: {len(raw_train)}, Val: {len(raw_val)}, Test: {len(raw_test)}")

# Tokenizers and models
logger.info("Loading tokenizers and models...")
enc_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_ENCODER)
dec_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_DECODER)
enc_model = AutoModel.from_pretrained(MODEL_NAME_ENCODER).to(device)
enc_model.eval()
dec_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_DECODER).to(device)

# Dataset class
class GETSumDataset(Dataset):
    def __init__(self, hf_dataset, encoder_tokenizer, decoder_tokenizer, max_sent_len=MAX_SENT_LEN, max_sents=MAX_DOC_SENT):
        self.ds = hf_dataset
        self.enc_tok = encoder_tokenizer
        self.dec_tok = decoder_tokenizer
        self.max_sent_len = max_sent_len
        self.max_sents = max_sents

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        article = item['article'] if item['article'] else ""
        summary = item['highlights'] if 'highlights' in item else item.get('summary', '')

        sents = split_sentences(article, max_sents=self.max_sents)
        encodings = [self.enc_tok(sent, truncation=True, max_length=self.max_sent_len, padding='max_length', return_tensors='pt') for sent in sents]
        input_ids = torch.stack([e['input_ids'].squeeze(0) for e in encodings]) if encodings else torch.zeros((0, self.max_sent_len), dtype=torch.long)
        attention_mask = torch.stack([e['attention_mask'].squeeze(0) for e in encodings]) if encodings else torch.zeros((0, self.max_sent_len), dtype=torch.long)
        dec_target = self.dec_tok(summary, truncation=True, max_length=256, padding='max_length', return_tensors='pt')

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'summary_ids': dec_target['input_ids'].squeeze(0),
            'summary_attention_mask': dec_target['attention_mask'].squeeze(0),
            'sent_texts': sents,
            'raw_summary': summary,
            'raw_article': article,
            'num_sents': input_ids.size(0)
        }

def collate_getsum(batch):
    max_sents = max(item['num_sents'] for item in batch)
    max_sent_len = batch[0]['input_ids'].size(1)
    all_input_ids = []
    all_att_masks = []
    sent_counts = []
    for item in batch:
        cnt = item['num_sents']
        sent_counts.append(cnt)
        if cnt < max_sents:
            pad_ids = torch.zeros((max_sents - cnt, max_sent_len), dtype=torch.long)
            pad_am = torch.zeros((max_sents - cnt, max_sent_len), dtype=torch.long)
            input_ids = torch.cat([item['input_ids'], pad_ids], dim=0)
            att_mask = torch.cat([item['attention_mask'], pad_am], dim=0)
        else:
            input_ids = item['input_ids']
            att_mask = item['attention_mask']
        all_input_ids.append(input_ids)
        all_att_masks.append(att_mask)
    return {
        'input_ids': torch.stack(all_input_ids),
        'attention_mask': torch.stack(all_att_masks),
        'summary_ids': torch.stack([item['summary_ids'] for item in batch]),
        'summary_attention_mask': torch.stack([item['summary_attention_mask'] for item in batch]),
        'sent_counts': torch.tensor(sent_counts),
        'sent_texts_list': [item['sent_texts'] for item in batch],
        'raw_summaries': [item['raw_summary'] for item in batch],
        'raw_articles': [item['raw_article'] for item in batch]
    }

# Data loaders
proc_train = GETSumDataset(raw_train, enc_tokenizer, dec_tokenizer)
train_loader = DataLoader(proc_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_getsum, num_workers=0, pin_memory=True)
proc_val = GETSumDataset(raw_val, enc_tokenizer, dec_tokenizer)
val_loader = DataLoader(proc_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_getsum, num_workers=0, pin_memory=True)
proc_test = GETSumDataset(raw_test, enc_tokenizer, dec_tokenizer)
test_loader = DataLoader(proc_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_getsum, num_workers=0, pin_memory=True)
logger.info(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

# Encoder utils
@torch.no_grad()
def encode_sentences_batch(input_ids_batch, attention_mask_batch, encoder_model):
    B, S, L = input_ids_batch.size()
    input_ids = input_ids_batch.view(B*S, L).to(device)
    attention_mask = attention_mask_batch.view(B*S, L).to(device)
    outputs = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden = outputs.last_hidden_state
    cls_emb = last_hidden[:,0,:]
    return cls_emb.view(B, S, -1)

# Graph layer
use_pyg = USE_PYG
try:
    import torch_geometric
    from torch_geometric.nn import GATConv
    use_pyg = True
    logger.info("Using PyG for GAT")
except ImportError:
    try:
        import dgl
        from dgl.nn import GATConv as DGLGATConv
        use_pyg = False
        logger.info("Using DGL for GAT")
    except ImportError:
        raise RuntimeError("Install PyG or DGL for GAT layers.")

class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        if use_pyg:
            self.gat1 = GATConv(in_dim, hid_dim // num_heads, heads=num_heads, dropout=dropout)
            self.gat2 = GATConv(hid_dim, hid_dim // num_heads, heads=num_heads, dropout=dropout)
        else:
            self.gat1 = DGLGATConv(in_dim, hid_dim // num_heads, num_heads)
            self.gat2 = DGLGATConv(hid_dim, hid_dim // num_heads, num_heads)
        self.elu = nn.ELU()

    def forward(self, sent_emb, adj_mask, sent_counts):
        B, S, H = sent_emb.size()
        node_feats = []
        edge_index_list = []
        offset = 0
        for b in range(B):
            cnt = int(sent_counts[b].item())
            feats = sent_emb[b, :cnt, :].contiguous()
            node_feats.append(feats)
            adj = adj_mask[b, :cnt, :cnt].detach().cpu().numpy()
            src, dst = np.where(adj == 1)
            edge_index_list.append((src + offset, dst + offset))
            offset += cnt
        if offset == 0:
            return torch.zeros((B, S, self.hid_dim), device=sent_emb.device)
        x = torch.cat(node_feats, dim=0).to(sent_emb.device)

        if use_pyg:
            srcs = np.concatenate([e[0] for e in edge_index_list]) if edge_index_list else np.array([], dtype=int)
            dsts = np.concatenate([e[1] for e in edge_index_list]) if edge_index_list else np.array([], dtype=int)
            edge_index = torch.tensor([srcs, dsts], dtype=torch.long).to(sent_emb.device)
            x1 = self.gat1(x, edge_index)
            x1 = x1.flatten(1) if x1.dim() == 3 else x1
            x2 = self.gat2(x1, edge_index)
            out = x2
        else:
            import dgl
            srcs = np.concatenate([e[0] for e in edge_index_list]) if edge_index_list else np.array([], dtype=int)
            dsts = np.concatenate([e[1] for e in edge_index_list]) if edge_index_list else np.array([], dtype=int)
            g = dgl.graph((srcs, dsts), num_nodes=offset).to(sent_emb.device)
            x1 = self.gat1(g, x)
            x1 = x1.flatten(1) if x1.dim() == 3 else x1
            x2 = self.gat2(g, x1)
            out = x2.flatten(1) if x2.dim() == 3 else x2

        hid_dim = out.size(1)
        batched = torch.zeros((B, S, hid_dim), device=sent_emb.device)
        ptr = 0
        for b in range(B):
            cnt = int(sent_counts[b].item())
            if cnt > 0:
                batched[b, :cnt, :] = out[ptr:ptr+cnt]
                ptr += cnt
        return batched

# GETSum Model
class GETSumModel(nn.Module):
    def __init__(self, enc_hidden_dim, gat_hidden_dim, dec_model_name=MODEL_NAME_DECODER, top_k=TOP_K):
        super().__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.gat_hidden_dim = gat_hidden_dim
        self.graph_encoder = GraphEncoder(enc_hidden_dim, gat_hidden_dim)
        self.scorer = nn.Linear(gat_hidden_dim, 1)
        self.gate = nn.Linear(enc_hidden_dim + gat_hidden_dim, gat_hidden_dim)
        self.top_k = top_k

    def forward_extractive(self, sent_emb, adj_mask, sent_counts):
        gat_h = self.graph_encoder(sent_emb, adj_mask, sent_counts)
        scores = self.scorer(gat_h).squeeze(-1)
        S = scores.size(1)
        running_idx = torch.arange(S).unsqueeze(0).to(device)
        valid_mask = (running_idx < sent_counts.unsqueeze(1)).to(device)
        # Use a smaller negative value for float16 compatibility
        scores = scores.masked_fill(~valid_mask, torch.tensor(-65504.0, dtype=scores.dtype, device=device))
        return scores, gat_h

    def fuse_representations(self, enc_sent_emb, gat_sent_emb):
        fused = torch.cat([enc_sent_emb, gat_sent_emb], dim=-1)
        gate = torch.sigmoid(self.gate(fused))
        enc_proj = nn.Linear(enc_sent_emb.size(-1), gat_sent_emb.size(-1)).to(device)
        enc_projed = enc_proj(enc_sent_emb)
        return gate * gat_sent_emb + (1-gate) * enc_projed

    def select_topk_sentences(self, scores, k):
        topk = torch.topk(scores, k=k, dim=1)
        return topk.indices

# Adjacency matrix
def build_adj_from_embeddings(sent_emb, sent_counts, threshold=0.3):
    B, S, H = sent_emb.size()
    sent_emb_cpu = sent_emb.detach().cpu().numpy()
    adj_masks = []
    for b in range(B):
        cnt = int(sent_counts[b].item())
        embs = sent_emb_cpu[b, :cnt, :]
        if cnt == 0:
            adj = np.zeros((S,S), dtype=np.uint8)
        else:
            sim = cosine_similarity(embs)
            adj_small = (sim >= threshold).astype(np.uint8)
            adj = np.zeros((S,S), dtype=np.uint8)
            adj[:cnt, :cnt] = adj_small
            np.fill_diagonal(adj, 1)
        adj_masks.append(torch.tensor(adj, dtype=torch.uint8))
    return torch.stack(adj_masks).to(device)

# Loss functions
def lcs(a, b):
    la, lb = len(a), len(b)
    dp = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la-1,-1,-1):
        for j in range(lb-1,-1,-1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i+1][j+1]
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j+1])
    return dp[0][0]

def rouge_l_score(sent, summary):
    s_tokens = sent.split()
    r_tokens = summary.split()
    if len(s_tokens) == 0 or len(r_tokens) == 0:
        return 0.0
    l = lcs(s_tokens, r_tokens)
    prec = l / len(s_tokens)
    rec = l / len(r_tokens)
    return (2*prec*rec)/(prec+rec) if prec + rec > 0 else 0.0

def make_oracle_labels(batch_raw_summaries, sent_lists, top_k=TOP_K):
    B = len(sent_lists)
    labels = []
    for b in range(B):
        sents = sent_lists[b]
        summary = batch_raw_summaries[b]
        scores = [rouge_l_score(s, summary) for s in sents]
        if len(scores) == 0:
            labels.append([])
            continue
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        labels.append(idxs)
    return labels

bce_loss = nn.BCEWithLogitsLoss()
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Training loop
model = GETSumModel(enc_hidden_dim=enc_model.config.hidden_size, gat_hidden_dim=enc_model.config.hidden_size//2, top_k=TOP_K).to(device)
if USE_DDP:
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    dec_model = DDP(dec_model, device_ids=[torch.cuda.current_device()])

optimizer = AdamW(list(model.parameters()) + list(dec_model.parameters()), lr=LEARNING_RATE)
total_steps = max(1, len(train_loader) * EPOCHS // ACCUM_STEPS)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
scaler = torch.amp.GradScaler('cuda', enabled=MIXED_PRECISION)

def evaluate_rouge(loader, model, dec_model, max_samples=None):
    model.eval()
    dec_model.eval()
    all_preds = []
    all_refs = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Evaluating")):
            if max_samples and i * BATCH_SIZE >= max_samples:
                break
            input_ids = batch['input_ids'].to(device)
            att_mask = batch['attention_mask'].to(device)
            sent_counts = batch['sent_counts'].to(device)
            sent_texts = batch['sent_texts_list']

            enc_sent_emb = encode_sentences_batch(input_ids, att_mask, enc_model)
            adj_mask = build_adj_from_embeddings(enc_sent_emb, sent_counts)
            scores, gat_h = model.forward_extractive(enc_sent_emb, adj_mask, sent_counts)
            topk_idxs = model.select_topk_sentences(scores, k=TOP_K)

            dec_inputs = []
            for b in range(input_ids.size(0)):
                sents = sent_texts[b][:int(sent_counts[b].item())]
                idxs = topk_idxs[b].cpu().tolist()
                chosen = [sents[i] if i < len(sents) else "" for i in idxs]
                dec_inputs.append(" ".join(chosen))
            dec_batch = dec_tokenizer(dec_inputs, truncation=True, padding=True, return_tensors='pt').to(device)
            gen = dec_model.generate(input_ids=dec_batch['input_ids'], attention_mask=dec_batch['attention_mask'], max_length=142, num_beams=4)
            summaries = dec_tokenizer.batch_decode(gen, skip_special_tokens=True)

            all_preds.extend(summaries)
            all_refs.extend(batch['raw_summaries'])

    scores = [scorer.score(ref, pred) for ref, pred in zip(all_refs, all_preds)]
    avg_rouge1 = np.mean([s['rouge1'].fmeasure for s in scores])
    avg_rouge2 = np.mean([s['rouge2'].fmeasure for s in scores])
    avg_rougeL = np.mean([s['rougeL'].fmeasure for s in scores])
    return avg_rouge1, avg_rouge2, avg_rougeL, all_preds, all_refs

logger.info("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    dec_model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    epoch_loss = 0.0
    step = 0
    optimizer.zero_grad()
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        att_mask = batch['attention_mask'].to(device)
        sent_counts = batch['sent_counts'].to(device)
        sent_texts = batch['sent_texts_list']

        B, S, L = input_ids.size()
        with torch.no_grad():
            enc_sent_emb = encode_sentences_batch(input_ids, att_mask, enc_model)

        adj_mask = build_adj_from_embeddings(enc_sent_emb, sent_counts)

        with torch.amp.autocast('cuda', enabled=MIXED_PRECISION):
            scores, gat_h = model.forward_extractive(enc_sent_emb, adj_mask, sent_counts)
            oracle_idxs = make_oracle_labels(batch['raw_summaries'], sent_texts, top_k=TOP_K)

            target_labels = torch.zeros_like(scores).to(device)
            for b in range(B):
                for idx in oracle_idxs[b]:
                    if idx < scores.size(1):
                        target_labels[b, idx] = 1.0

            loss_rank = bce_loss(scores, target_labels)

            topk_idxs = model.select_topk_sentences(scores, k=TOP_K)
            dec_inputs = []
            for b in range(B):
                idxs = topk_idxs[b].cpu().tolist()
                sents = sent_texts[b]
                chosen = [sents[i] if i < len(sents) else "" for i in idxs]
                dec_inputs.append(" ".join(chosen))

            dec_batch = dec_tokenizer(dec_inputs, truncation=True, padding=True, return_tensors='pt').to(device)
            target_ids = batch['summary_ids'].to(device)

            outputs = dec_model(input_ids=dec_batch['input_ids'], attention_mask=dec_batch['attention_mask'], labels=target_ids)
            loss_gen = outputs.loss

            loss = (W_RANK * loss_rank + W_GEN * loss_gen) / ACCUM_STEPS

        scaler.scale(loss).backward()
        step += 1
        if step % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        epoch_loss += loss.item() * ACCUM_STEPS
        loop.set_postfix({'loss': loss.item() * ACCUM_STEPS})

    logger.info(f"Epoch {epoch+1} avg train loss: {epoch_loss/len(train_loader):.4f}")

    # Validation
    r1, r2, rl, _, _ = evaluate_rouge(val_loader, model.module if USE_DDP else model, dec_model.module if USE_DDP else dec_model, max_samples=1000)
    logger.info(f"Epoch {epoch+1} val ROUGE-1: {r1:.4f}, ROUGE-2: {r2:.4f}, ROUGE-L: {rl:.4f}")

    # Checkpoint
    save_dir = f'C:\\Users\\PC\\Desktop\\IRNLP PROJECT\\checkpoints\\epoch{epoch+1}'
    os.makedirs(save_dir, exist_ok=True)
    if USE_DDP:
        dec_model.module.save_pretrained(os.path.join(save_dir, 'bart_decoder'))
        torch.save(model.module.state_dict(), os.path.join(save_dir, 'getsum_model.pt'))
    else:
        dec_model.save_pretrained(os.path.join(save_dir, 'bart_decoder'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'getsum_model.pt'))
    logger.info(f"Checkpoint saved to {save_dir}")

# Test inference
logger.info("Running inference on test set...")
r1, r2, rl, test_preds, test_refs = evaluate_rouge(test_loader, model.module if USE_DDP else model, dec_model.module if USE_DDP else dec_model)
logger.info(f"Test ROUGE-1: {r1:.4f}, ROUGE-2: {r2:.4f}, ROUGE-L: {rl:.4f}")

# Save predictions
save_dir = 'C:\\Users\\PC\\Desktop\\IRNLP PROJECT\\checkpoints'
os.makedirs(save_dir, exist_ok=True)
import json
with open(os.path.join(save_dir, 'test_predictions.json'), 'w') as f:
    json.dump({'predictions': test_preds, 'references': test_refs}, f, indent=2)
logger.info("Test predictions saved.")

logger.info("Examples:")
for i in range(min(5, len(test_preds))):
    logger.info(f"REF: {test_refs[i]}")
    logger.info(f"PRED: {test_preds[i]}")
    logger.info('---')