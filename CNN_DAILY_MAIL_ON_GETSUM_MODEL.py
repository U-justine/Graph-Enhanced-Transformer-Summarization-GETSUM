import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup, GenerationConfig
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm.auto import tqdm
import numpy as np
import logging
import json
import spacy
from torch_geometric.nn import GATConv
from torch_geometric.data import Batch, Data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
MODEL_NAME_ENCODER = "bert-base-uncased"
MODEL_NAME_DECODER = "facebook/bart-base"
MAX_SENT_LEN = 128
MAX_DOC_SENT = 512  # Increased to minimize truncation
TOP_K = 3
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
EPOCHS = 5
W_RANK = 0.5
W_GEN = 1.5
USE_DDP = False
MIXED_PRECISION = True
ACCUM_STEPS = 4
PATIENCE = 2
COSINE_THRESHOLD = 0.4
GRAD_CLIP_NORM = 1.0
LOCAL_DATA_DIR = r"C:\Users\PC\Desktop\IRNLP PROJECT\cnn_dailymail"
CACHE_DIR = os.path.expanduser(r"~/.cache/huggingface/datasets")
CHECKPOINT_DIR = r"C:\Users\PC\Desktop\IRNLP PROJECT\checkpoints"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------- Helpers ----------
def split_sentences(text, max_sents=MAX_DOC_SENT):
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        doc = nlp(text)
        sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        orig_len = len(sents)
        if orig_len > max_sents:
            sents = sorted(sents, key=lambda x: len(x.split()), reverse=True)[:max_sents]
            logger.info(f"Article truncated from {orig_len} to {max_sents} sentences")
        return sents
    except Exception as e:
        logger.error(f"Sentence splitting failed: {e}")
        return []

# ---------- Dataset class ----------
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
        article = item.get('article', "") or ""
        summary = item.get('highlights', "") or ""
        if not article.strip() or not summary.strip():
            return self._dummy_item(summary)
        sents = split_sentences(article, max_sents=self.max_sents)
        if not sents:
            return self._dummy_item(summary)

        encodings = [self.enc_tok(sent, truncation=True, max_length=self.max_sent_len, padding='max_length', return_tensors='pt') for sent in sents]
        input_ids = torch.stack([e['input_ids'].squeeze(0) for e in encodings])
        attention_mask = torch.stack([e['attention_mask'].squeeze(0) for e in encodings])
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

    def _dummy_item(self, summary):
        dec_target = self.dec_tok(summary, truncation=True, max_length=256, padding='max_length', return_tensors='pt')
        return {
            'input_ids': torch.zeros((0, self.max_sent_len), dtype=torch.long),
            'attention_mask': torch.zeros((0, self.max_sent_len), dtype=torch.long),
            'summary_ids': dec_target['input_ids'].squeeze(0),
            'summary_attention_mask': dec_target['attention_mask'].squeeze(0),
            'sent_texts': [],
            'raw_summary': summary,
            'raw_article': '',
            'num_sents': 0
        }

def collate_getsum(batch):
    max_sents = max(item['num_sents'] for item in batch)
    max_sent_len = max((item['input_ids'].size(1) if item['num_sents'] > 0 else MAX_SENT_LEN) for item in batch)
    all_input_ids = []
    all_att_masks = []
    sent_counts = []
    for item in batch:
        cnt = item['num_sents']
        sent_counts.append(cnt)
        if cnt == 0:
            input_ids = torch.zeros((max_sents, max_sent_len), dtype=torch.long)
            att_mask = torch.zeros((max_sents, max_sent_len), dtype=torch.long)
        else:
            cur_ids = item['input_ids']
            cur_am = item['attention_mask']
            if cur_ids.size(1) < max_sent_len:
                pad_len = max_sent_len - cur_ids.size(1)
                pad_ids = torch.zeros((cnt, pad_len), dtype=torch.long)
                pad_am = torch.zeros((cnt, pad_len), dtype=torch.long)
                cur_ids = torch.cat([cur_ids, pad_ids], dim=1)
                cur_am = torch.cat([cur_am, pad_am], dim=1)
            if cnt < max_sents:
                pad_ids = torch.zeros((max_sents - cnt, max_sent_len), dtype=torch.long)
                pad_am = torch.zeros((max_sents - cnt, max_sent_len), dtype=torch.long)
                input_ids = torch.cat([cur_ids, pad_ids], dim=0)
                att_mask = torch.cat([cur_am, pad_am], dim=0)
            else:
                input_ids = cur_ids
                att_mask = cur_am
        all_input_ids.append(input_ids)
        all_att_masks.append(att_mask)

    summary_ids = torch.stack([item['summary_ids'] for item in batch])
    summary_attention_mask = torch.stack([item['summary_attention_mask'] for item in batch])

    return {
        'input_ids': torch.stack(all_input_ids),
        'attention_mask': torch.stack(all_att_masks),
        'summary_ids': summary_ids,
        'summary_attention_mask': summary_attention_mask,
        'sent_counts': torch.tensor(sent_counts, dtype=torch.long),
        'sent_texts_list': [item['sent_texts'] for item in batch],
        'raw_summaries': [item['raw_summary'] for item in batch],
        'raw_articles': [item['raw_article'] for item in batch]
    }

# ---------- Encoder utility ----------
@torch.no_grad()
def encode_sentences_batch(input_ids_batch, attention_mask_batch, encoder_model):
    B, S, L = input_ids_batch.size()
    input_ids = input_ids_batch.view(B * S, L).to(device)
    attention_mask = attention_mask_batch.view(B * S, L).to(device)
    outputs = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden = outputs.last_hidden_state
    cls_emb = last_hidden[:, 0, :].contiguous()
    cls_emb = torch.where(torch.isfinite(cls_emb), cls_emb, torch.zeros_like(cls_emb))
    return cls_emb.view(B, S, -1)

# ---------- PyG Graph layer ----------
class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_dim, hid_dim // 1, heads=num_heads, concat=False, dropout=dropout)
        self.gat2 = GATConv(hid_dim, hid_dim // 1, heads=1, concat=False, dropout=dropout)
        self.elu = nn.ELU()
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, sent_emb, edge_index_mask, sent_counts):
        B, S, H = sent_emb.size()
        if torch.isnan(sent_emb).any() or torch.isinf(sent_emb).any():
            logger.info("NaN/Inf in sent_emb. Replacing with zeros.")
            sent_emb = torch.where(torch.isfinite(sent_emb), sent_emb, torch.zeros_like(sent_emb))
        node_feats = []
        edge_indices = []
        batch_idx = []
        offset = 0
        for b in range(B):
            cnt = int(sent_counts[b].item())
            if cnt == 0:
                continue
            feats = sent_emb[b, :cnt, :].contiguous()
            node_feats.append(feats)
            batch_idx.extend([b] * cnt)
            mask = edge_index_mask[b, :cnt, :cnt].to(torch.bool)
            if mask.sum() == 0:
                srcs = torch.arange(cnt, device=feats.device)
                dsts = torch.arange(cnt, device=feats.device)
            else:
                srcs, dsts = mask.nonzero(as_tuple=True)
            edge_indices.append(torch.stack([srcs + offset, dsts + offset], dim=0))
            offset += cnt

        if offset == 0:
            return torch.zeros((B, S, self.gat2.out_channels), device=sent_emb.device)

        x = torch.cat(node_feats, dim=0)
        edge_index = torch.cat(edge_indices, dim=1).to(device)
        batch_tensor = torch.tensor(batch_idx, dtype=torch.long).to(device)

        data = Batch(x=x, edge_index=edge_index, batch=batch_tensor)
        x1 = self.elu(self.gat1(data.x, data.edge_index))
        x1 = self.norm(x1)
        x2 = self.elu(self.gat2(x1, data.edge_index))
        x2 = self.norm(x2)
        x2 = torch.clamp(x2, -1e4, 1e4)
        out = x2

        hid_dim = out.size(1)
        batched = torch.zeros((B, S, hid_dim), device=sent_emb.device)
        ptr = 0
        for b in range(B):
            cnt = int(sent_counts[b].item())
            if cnt > 0:
                batched[b, :cnt, :] = out[ptr:ptr + cnt]
                ptr += cnt
        return batched

# ---------- GETSum Model ----------
class GETSumModel(nn.Module):
    def __init__(self, enc_hidden_dim, gat_hidden_dim, top_k=TOP_K):
        super().__init__()
        self.graph_encoder = GraphEncoder(enc_hidden_dim, gat_hidden_dim)
        self.scorer = nn.Linear(gat_hidden_dim, 1)
        self.gate = nn.Linear(enc_hidden_dim + gat_hidden_dim, gat_hidden_dim)
        self.enc_proj = nn.Linear(enc_hidden_dim, gat_hidden_dim)
        self.top_k = top_k

    def forward_extractive(self, sent_emb, adj_mask, sent_counts):
        gat_h = self.graph_encoder(sent_emb, adj_mask, sent_counts)
        scores = self.scorer(gat_h).squeeze(-1)
        running_idx = torch.arange(scores.size(1), device=device).unsqueeze(0).expand(scores.size(0), -1)
        valid_mask = running_idx < sent_counts.unsqueeze(1)
        scores = scores.masked_fill(~valid_mask, float('-inf'))
        scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
        return scores, gat_h

    def fuse_representations(self, enc_sent_emb, gat_sent_emb):
        fused = torch.cat([enc_sent_emb, gat_sent_emb], dim=-1)
        gate = torch.sigmoid(self.gate(fused))
        enc_projed = self.enc_proj(enc_sent_emb)
        return gate * gat_sent_emb + (1 - gate) * enc_projed

    def select_topk_sentences(self, scores, k):
        k = max(1, min(k, scores.size(1)))
        topk = torch.topk(scores, k=k, dim=1)
        return topk.indices

# ---------- Adjacency builder ----------
def build_adj_from_embeddings(sent_emb, sent_counts, threshold=COSINE_THRESHOLD):
    B, S, H = sent_emb.size()
    adj_masks = []
    for b in range(B):
        cnt = int(sent_counts[b].item())
        if cnt == 0:
            adj = torch.zeros((S, S), dtype=torch.bool, device=device)
        else:
            embs = sent_emb[b, :cnt, :].to(device)
            if torch.isnan(embs).any() or torch.isinf(embs).any():
                logger.info(f"NaN/Inf in embeddings for batch {b}. Replacing with zeros.")
                embs = torch.where(torch.isfinite(embs), embs, torch.zeros_like(embs))
            norm = torch.norm(embs, dim=1, keepdim=True)
            norm = torch.where(norm > 1e-8, norm, torch.ones_like(norm))
            embs_norm = embs / norm
            sim = torch.mm(embs_norm, embs_norm.t())
            sim = torch.where(torch.isfinite(sim), sim, torch.zeros_like(sim))
            adj_small = (sim >= threshold)
            adj = torch.zeros((S, S), dtype=torch.bool, device=device)
            adj[:cnt, :cnt] = adj_small
            idxs = torch.arange(cnt, device=device)
            adj[idxs, idxs] = True
        adj_masks.append(adj)
    return torch.stack(adj_masks, dim=0)

# ---------- Oracle labels and metrics ----------
def lcs(a, b):
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la):
        for j in range(lb):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = 1 + dp[i][j]
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[la][lb]

def rouge_l_score(sent, summary):
    s_tokens = sent.split()
    r_tokens = summary.split()
    if not s_tokens or not r_tokens:
        return 0.0
    l = lcs(s_tokens, r_tokens)
    prec = l / len(s_tokens)
    rec = l / len(r_tokens)
    return (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0.0

def make_oracle_labels(batch_raw_summaries, sent_lists, top_k=TOP_K):
    B = len(sent_lists)
    labels = []
    for b in range(B):
        sents = sent_lists[b]
        summary = batch_raw_summaries[b]
        if not sents or not summary.strip():
            labels.append([])
            continue
        scores = [rouge_l_score(s, summary) for s in sents]
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        labels.append([i for i in idxs if i < len(sents)])
    return labels

bce_loss = nn.BCEWithLogitsLoss()
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# ---------- Evaluation ----------
def evaluate_rouge(loader, model, dec_model, max_samples=None):
    model.eval()
    dec_model.eval()
    all_preds = []
    all_refs = []
    generation_config = GenerationConfig(
        max_new_tokens=142,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        forced_bos_token_id=0
    )
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Evaluating")):
            if max_samples and i * loader.batch_size >= max_samples:
                break
            input_ids = batch['input_ids'].to(device)
            att_mask = batch['attention_mask'].to(device)
            sent_counts = batch['sent_counts'].to(device)
            sent_texts = batch['sent_texts_list']
            if sent_counts.sum() == 0:
                continue

            enc_sent_emb = encode_sentences_batch(input_ids, att_mask, enc_model)
            adj_mask = build_adj_from_embeddings(enc_sent_emb, sent_counts)
            scores, gat_h = model.forward_extractive(enc_sent_emb, adj_mask, sent_counts)
            topk_idxs = model.select_topk_sentences(scores, k=TOP_K)

            dec_inputs = []
            for b in range(input_ids.size(0)):
                sents = sent_texts[b][:int(sent_counts[b].item())]
                idxs = topk_idxs[b].cpu().tolist()
                chosen = [sents[i] if (i < len(sents) and i >= 0) else "" for i in idxs]
                dec_input = " ".join([c for c in chosen if c]).strip()
                if not dec_input:
                    dec_input = sents[0] if sents else "<empty>"
                dec_inputs.append(dec_input)
            dec_batch = dec_tokenizer(dec_inputs, truncation=True, padding=True, max_length=256, return_tensors='pt').to(device)
            gen = dec_model.generate(
                input_ids=dec_batch['input_ids'],
                attention_mask=dec_batch['attention_mask'],
                generation_config=generation_config
            )
            summaries = dec_tokenizer.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            all_preds.extend(summaries)
            all_refs.extend(batch['raw_summaries'])

    if not all_preds or not all_refs:
        return 0.0, 0.0, 0.0, [], []
    scores_list = [scorer.score(ref, pred) for ref, pred in zip(all_refs, all_preds)]
    avg_rouge1 = np.mean([s['rouge1'].fmeasure for s in scores_list])
    avg_rouge2 = np.mean([s['rouge2'].fmeasure for s in scores_list])
    avg_rougeL = np.mean([s['rougeL'].fmeasure for s in scores_list])
    return avg_rouge1, avg_rouge2, avg_rougeL, all_preds, all_refs

if __name__ == '__main__':
    # ---------- Device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        try:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        except Exception:
            pass

    # ---------- spaCy ----------
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        logger.info(f"spaCy pipeline: {nlp.pipe_names}")
    except Exception as e:
        logger.error("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
        raise

    # ---------- Load dataset ----------
    logger.info("Loading dataset...")
    dataset = None
    try:
        train_fp = os.path.join(LOCAL_DATA_DIR, "train.csv")
        val_fp = os.path.join(LOCAL_DATA_DIR, "validation.csv")
        test_fp = os.path.join(LOCAL_DATA_DIR, "test.csv")
        if os.path.exists(train_fp) and os.path.exists(val_fp) and os.path.exists(test_fp):
            logger.info(f"Loading from local CSVs in {LOCAL_DATA_DIR}...")
            dataset = {
                'train': load_dataset("csv", data_files=train_fp, cache_dir=CACHE_DIR)['train'],
                'validation': load_dataset("csv", data_files=val_fp, cache_dir=CACHE_DIR)['train'],
                'test': load_dataset("csv", data_files=test_fp, cache_dir=CACHE_DIR)['train'],
            }
        else:
            logger.info("Local CSVs not found. Downloading from Hugging Face...")
            dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir=CACHE_DIR)
            dataset = {
                'train': dataset['train'],
                'validation': dataset['validation'],
                'test': dataset['test']
            }
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    for split_name, ds in dataset.items():
        cols = ds.column_names
        if 'article' not in cols or 'highlights' not in cols:
            raise ValueError(f"Split '{split_name}' missing required columns 'article' and 'highlights'. Found: {cols}")

    dataset = {
        split: ds.filter(lambda x: x['article'] is not None and isinstance(x['article'], str) and x['article'].strip() and
                         x['highlights'] is not None and isinstance(x['highlights'], str) and x['highlights'].strip())
        for split, ds in dataset.items()
    }
    raw_train = dataset['train']
    raw_val = dataset['validation']
    raw_test = dataset['test']
    logger.info(f"After filtering: Train={len(raw_train)}, Val={len(raw_val)}, Test={len(raw_test)}")

    # Profile article lengths
    for split_name, ds in dataset.items():
        lengths = [len(split_sentences(item['article'])) for item in ds]
        trunc_count = sum(1 for l in lengths if l > MAX_DOC_SENT)
        logger.info(f"{split_name.capitalize()} article length stats: mean={np.mean(lengths):.1f}, max={max(lengths)}, "
                    f"95th percentile={np.percentile(lengths, 95):.1f}, truncated articles={trunc_count} ({trunc_count/len(lengths)*100:.2f}%)")

    # ---------- Tokenizers & models ----------
    logger.info("Loading tokenizers and models...")
    enc_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_ENCODER)
    dec_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_DECODER)
    enc_model = AutoModel.from_pretrained(MODEL_NAME_ENCODER).to(device)
    enc_model.eval()
    dec_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_DECODER).to(device)

    # ---------- DataLoaders ----------
    num_workers = 0  # Keep at 0 for stability; increase later if needed
    proc_train = GETSumDataset(raw_train, enc_tokenizer, dec_tokenizer)
    train_loader = DataLoader(proc_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_getsum, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    proc_val = GETSumDataset(raw_val, enc_tokenizer, dec_tokenizer)
    val_loader = DataLoader(proc_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_getsum, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    proc_test = GETSumDataset(raw_test, enc_tokenizer, dec_tokenizer)
    test_loader = DataLoader(proc_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_getsum, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    logger.info(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # ---------- Training setup ----------
    model = GETSumModel(enc_hidden_dim=enc_model.config.hidden_size, gat_hidden_dim=enc_model.config.hidden_size // 2, top_k=TOP_K).to(device)
    optimizer = AdamW(list(model.parameters()) + list(dec_model.parameters()), lr=LEARNING_RATE)
    num_train_steps = max(1, math.ceil(len(train_loader) * EPOCHS / max(1, ACCUM_STEPS)))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_train_steps), num_training_steps=num_train_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and device.type == "cuda"))

    logger.info("Starting training...")
    best_rouge = 0.0
    no_improve = 0

    try:
        for epoch in range(EPOCHS):
            model.train()
            dec_model.train()
            epoch_loss = 0.0
            step = 0
            optimizer.zero_grad()
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            for batch in loop:
                input_ids = batch['input_ids'].to(device)
                att_mask = batch['attention_mask'].to(device)
                sent_counts = batch['sent_counts'].to(device)
                sent_texts = batch['sent_texts_list']
                bsize = input_ids.size(0)
                if sent_counts.sum() == 0:
                    logger.info("Skipping batch with all empty articles")
                    continue

                # Monitor GPU memory
                if device.type == "cuda":
                    mem = torch.cuda.memory_allocated() / 1024**3
                    logger.debug(f"Step {step}: GPU memory used: {mem:.2f} GB")

                with torch.no_grad():
                    enc_sent_emb = encode_sentences_batch(input_ids, att_mask, enc_model)
                    if torch.isnan(enc_sent_emb).any() or torch.isinf(enc_sent_emb).any():
                        logger.info("NaN/Inf in enc_sent_emb. Skipping batch.")
                        continue

                adj_mask = build_adj_from_embeddings(enc_sent_emb, sent_counts)

                with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and device.type == "cuda")):
                    scores, gat_h = model.forward_extractive(enc_sent_emb, adj_mask, sent_counts)
                    if torch.isnan(scores).any() or torch.isinf(scores).any():
                        logger.info(f"NaN/Inf in scores at step {step}. Skipping batch.")
                        optimizer.zero_grad()
                        continue

                    oracle_idxs = make_oracle_labels(batch['raw_summaries'], sent_texts, top_k=TOP_K)
                    target_labels = torch.zeros_like(scores).to(device)
                    for b in range(bsize):
                        for idx in oracle_idxs[b]:
                            if 0 <= idx < scores.size(1):
                                target_labels[b, idx] = 1.0

                    loss_rank = bce_loss(scores.float(), target_labels.float())
                    topk_idxs = model.select_topk_sentences(scores, k=TOP_K)
                    dec_inputs = []
                    for b in range(bsize):
                        idxs = topk_idxs[b].cpu().tolist()
                        sents = sent_texts[b]
                        chosen = [sents[i] if (i < len(sents) and i >= 0) else "" for i in idxs]
                        dec_input = " ".join([c for c in chosen if c]).strip()
                        if not dec_input:
                            dec_input = sents[0] if sents else "<empty>"
                            logger.info(f"Empty decoder input for batch {b}, using first sentence")
                        dec_inputs.append(dec_input)

                    dec_batch = dec_tokenizer(dec_inputs, truncation=True, padding=True, max_length=256, return_tensors='pt').to(device)
                    target_ids = batch['summary_ids'].to(device)
                    outputs = dec_model(input_ids=dec_batch['input_ids'], attention_mask=dec_batch['attention_mask'], labels=target_ids)
                    loss_gen = outputs.loss
                    loss = (W_RANK * loss_rank + W_GEN * loss_gen) / max(1, ACCUM_STEPS)

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.info(f"NaN/Inf loss at step {step}: loss_rank={loss_rank.item()}, loss_gen={loss_gen.item()}. Skipping batch.")
                    optimizer.zero_grad()
                    continue

                scaler.scale(loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(dec_model.parameters()), GRAD_CLIP_NORM)
                step += 1
                if step % ACCUM_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                epoch_loss += loss.item() * max(1, ACCUM_STEPS)
                loop.set_postfix({'loss': loss.item() * max(1, ACCUM_STEPS), 'rank': loss_rank.item(), 'gen': loss_gen.item(), 'grad_norm': grad_norm})

                # Checkpoint every 50 steps
                if step % 50 == 0:
                    checkpoint_dir = os.path.join(CHECKPOINT_DIR, f'epoch{epoch+1}_step{step}')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    dec_model.save_pretrained(os.path.join(checkpoint_dir, 'bart_decoder'))
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'getsum_model.pt'))
                    logger.info(f"Checkpoint saved at step {step} to {checkpoint_dir}")

            avg_loss = epoch_loss / max(1, len(train_loader))
            logger.info(f"Epoch {epoch+1} avg train loss: {avg_loss:.4f}")

            r1, r2, rl, _, _ = evaluate_rouge(val_loader, model, dec_model)
            logger.info(f"Epoch {epoch+1} val ROUGE-1: {r1:.4f}, ROUGE-2: {r2:.4f}, ROUGE-L: {rl:.4f}")

            if r1 > best_rouge:
                best_rouge = r1
                no_improve = 0
                checkpoint_dir = os.path.join(CHECKPOINT_DIR, f'epoch{epoch+1}')
                os.makedirs(checkpoint_dir, exist_ok=True)
                dec_model.save_pretrained(os.path.join(checkpoint_dir, 'bart_decoder'))
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'getsum_model.pt'))
                logger.info(f"Best checkpoint saved to {checkpoint_dir}")
            else:
                no_improve += 1

            if no_improve >= PATIENCE:
                logger.info("Early stopping triggered")
                break

    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving current model state...")
        checkpoint_dir = os.path.join(CHECKPOINT_DIR, 'interrupted')
        os.makedirs(checkpoint_dir, exist_ok=True)
        dec_model.save_pretrained(os.path.join(checkpoint_dir, 'bart_decoder'))
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'getsum_model.pt'))
        logger.info(f"Interrupted checkpoint saved to {checkpoint_dir}")
        exit(0)

    # ---------- Test inference ----------
    logger.info("Running inference on test set...")
    r1, r2, rl, test_preds, test_refs = evaluate_rouge(test_loader, model, dec_model)
    logger.info(f"Test ROUGE-1: {r1:.4f}, ROUGE-2: {r2:.4f}, ROUGE-L: {rl:.4f}")

    try:
        with open(os.path.join(CHECKPOINT_DIR, 'test_predictions.json'), 'w', encoding='utf-8') as f:
            json.dump({'predictions': test_preds, 'references': test_refs}, f, indent=2, ensure_ascii=False)
        logger.info("Test predictions saved.")
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")

    logger.info("Examples:")
    for i in range(min(5, len(test_preds))):
        logger.info(f"REF: {test_refs[i]}")
        logger.info(f"PRED: {test_preds[i]}")
        logger.info('---')
