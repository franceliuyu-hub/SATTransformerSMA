"""version 2"""
"""
╔══════════════════════════════════════════════════════════════════════╗
║        MAS-SAT : Système Multi-Agent pour la résolution SAT    ║
║        Kaggle — 2× NVIDIA Tesla T4                             ║
╠══════════════════════════════════════════════════════════════════════╣
║  Agents                                                        ║
║    A0 — Orchestrateur    : coordonne le pipeline                    ║
║    A1 — Préprocesseur    : tokenisation et analyse structurelle      ║
║    A2 — Solveur exact    : résolution PySAT (ground truth)          ║
║    A3 — Transformer      : classification neuronale sur GPU         ║
║    A4 — Évaluateur       : métriques et rapport graphique           ║
║                                                                     ║
║  Communication : MessageBus synchrone (publish / consume)          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ── Imports ────────────────────────────────────────────────────────────
import os, time, copy, random, gc, warnings
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from pysat.formula import CNF
from pysat.solvers import Solver as SATSolver

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score,
)

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

SEED       = 42
MAX_LEN    = 512
EPOCHS     = 25
BATCH_SIZE = 16
PATIENCE   = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# Détection GPU
_n_gpu = torch.cuda.device_count()
DEVICE = torch.device('cuda:0' if _n_gpu >= 1 else 'cpu')
USE_DP = _n_gpu >= 2          # DataParallel sur les deux T4

if _n_gpu >= 2:
    print(f"[GPU] {_n_gpu} GPU détectés : {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_name(1)}")
elif _n_gpu == 1:
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
else:
    print("[GPU] CPU mode")


# ══════════════════════════════════════════════════════════════════════
#  BUS DE COMMUNICATION
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Message:
    sender:    str
    receiver:  str
    msg_type:  str           # 'task' | 'result' | 'error'
    payload:   Any = None
    timestamp: float = field(default_factory=time.time)


class MessageBus:
    """File de messages par destinataire. Interface publish / consume."""

    def __init__(self):
        self._queues:   Dict[str, List[Message]] = defaultdict(list)
        self._history:  List[Message]            = []
        self._counters: Dict[str, int]           = defaultdict(int)

    def publish(self, msg: Message):
        self._queues[msg.receiver].append(msg)
        self._history.append(msg)
        self._counters[f"{msg.sender}→{msg.receiver}"] += 1

    def consume(self, agent_id: str) -> List[Message]:
        msgs = self._queues.pop(agent_id, [])
        return msgs

    @property
    def stats(self) -> Dict:
        return {
            'total':    len(self._history),
            'channels': dict(self._counters),
        }


# ══════════════════════════════════════════════════════════════════════
#  AGENT DE BASE
# ══════════════════════════════════════════════════════════════════════

class Agent(ABC):
    """Contrat commun : identité, accès au bus, journalisation, chrono."""

    def __init__(self, agent_id: str, bus: MessageBus):
        self.agent_id = agent_id
        self.bus      = bus
        self._elapsed = 0.0

    def log(self, msg: str):
        print(f"  [{self.agent_id}] {msg}")

    def publish(self, receiver: str, msg_type: str, payload: Any = None):
        self.bus.publish(Message(self.agent_id, receiver, msg_type, payload))

    def consume(self) -> List[Message]:
        return self.bus.consume(self.agent_id)

    @property
    def elapsed(self) -> float:
        return self._elapsed

    @abstractmethod
    def run(self, *args, **kwargs) -> Dict:
        ...


# ══════════════════════════════════════════════════════════════════════
#  CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════════

def load_satlib(directory: str, max_instances: int) -> List[CNF]:
    """Charge des fichiers .cnf depuis un répertoire SATLIB."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Répertoire introuvable : {directory}")

    files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory) if f.endswith('.cnf')
    ])[:max_instances]

    if not files:
        raise FileNotFoundError(f"Aucun fichier .cnf trouvé dans {directory}")

    instances = []
    for path in files:
        try:
            f = CNF(from_file=path)
            f.path = path
            instances.append(f)
        except Exception as e:
            print(f"  [WARN] Lecture échouée {path}: {e}")

    print(f"  ✓ {len(instances)} instances SATLIB chargées depuis {directory}")
    return instances


def generate_fallback(n_vars: int, n_clauses: int,
                      n_sat: int, n_unsat: int) -> List[CNF]:
    """
    Génère des instances 3-SAT synthétiques équilibrées.
    Utilisé uniquement si SATLIB est déséquilibré ou absent.
    """
    print("  [WARN] Fallback : génération synthétique")
    random.seed(SEED); np.random.seed(SEED)

    def rand_clause(nv: int) -> List[int]:
        vs = random.sample(range(1, nv + 1), 3)
        return [v * random.choice([-1, 1]) for v in vs]

    instances, sat_c, unsat_c = [], 0, 0
    max_tries = (n_sat + n_unsat) * 30

    for _ in range(max_tries):
        if sat_c >= n_sat and unsat_c >= n_unsat:
            break
        clauses = [rand_clause(n_vars) for _ in range(n_clauses)]
        with SATSolver(name='g3') as s:
            s.append_formula(clauses)
            is_sat = s.solve()

        if is_sat and sat_c < n_sat:
            f = CNF(); f.clauses = clauses; f.nv = n_vars
            f.path = f"syn_sat_{sat_c}.cnf"; f._label = 1
            instances.append(f); sat_c += 1
        elif not is_sat and unsat_c < n_unsat:
            f = CNF(); f.clauses = clauses; f.nv = n_vars
            f.path = f"syn_unsat_{unsat_c}.cnf"; f._label = 0
            instances.append(f); unsat_c += 1

    print(f"  ✓ Synthétique — SAT={sat_c}  UNSAT={unsat_c}")
    return instances


# ══════════════════════════════════════════════════════════════════════
#  AGENT A1 — PRÉPROCESSEUR
# ══════════════════════════════════════════════════════════════════════

SPECIAL = ['PAD', 'AND', 'OR', 'NOT']

class PreprocessorAgent(Agent):
    """
    Construit le vocabulaire, convertit chaque formule CNF en séquence
    d'ids padded, et extrait des statistiques structurelles.
    """

    def _build_vocab(self, instances: List[CNF]) -> Dict[str, int]:
        vars_set = set()
        for f in instances:
            for clause in f.clauses:
                for lit in clause:
                    vars_set.add(abs(lit))
        token2id = {t: i for i, t in enumerate(SPECIAL)}
        for v in sorted(vars_set):
            token2id[f'x{v}'] = len(token2id)
        return token2id

    def _to_ids(self, formula: CNF, token2id: Dict[str, int]) -> torch.Tensor:
        tokens = []
        for i, clause in enumerate(formula.clauses):
            for j, lit in enumerate(clause):
                if lit < 0:
                    tokens.append('NOT')
                tokens.append(f'x{abs(lit)}')
                if j < len(clause) - 1:
                    tokens.append('OR')
            if i < len(formula.clauses) - 1:
                tokens.append('AND')
        pad = token2id['PAD']
        ids = [token2id.get(t, pad) for t in tokens][:MAX_LEN]
        ids += [pad] * (MAX_LEN - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def _struct_stats(self, formula: CNF) -> Dict:
        n_lit = sum(len(c) for c in formula.clauses)
        neg   = sum(1 for c in formula.clauses for l in c if l < 0)
        return {
            'n_clauses':  len(formula.clauses),
            'n_vars':     formula.nv,
            'ratio':      len(formula.clauses) / max(1, formula.nv),
            'avg_clause': n_lit / max(1, len(formula.clauses)),
            'neg_ratio':  neg / max(1, n_lit),
        }

    def run(self, instances: List[CNF]) -> Dict:
        t0 = time.time()
        self.log(f"Tokenisation de {len(instances)} formules")

        token2id   = self._build_vocab(instances)
        tokenized  = [self._to_ids(f, token2id) for f in instances]
        stats_list = [self._struct_stats(f)      for f in instances]

        self._elapsed = time.time() - t0
        self.log(f"Vocabulaire : {len(token2id)} tokens — {self._elapsed:.2f}s")

        result = {
            'token2id':   token2id,
            'vocab_size': len(token2id),
            'tokenized':  tokenized,
            'stats_list': stats_list,
        }
        self.publish('A0', 'result', result)
        return result


# ══════════════════════════════════════════════════════════════════════
#  AGENT A2 — SOLVEUR EXACT (PySAT)
# ══════════════════════════════════════════════════════════════════════

class SymbolicSolverAgent(Agent):
    """
    Résout chaque formule CNF avec Glucose3.
    Fournit les labels ground truth (SAT=1, UNSAT=0)
    et les temps de résolution par instance.
    """

    def run(self, instances: List[CNF]) -> Dict:
        t0 = time.time()
        self.log(f"Résolution exacte de {len(instances)} formules (Glucose3)")

        labels, times = [], []
        for f in instances:
            if hasattr(f, '_label'):
                labels.append(f._label)
                times.append(0.0)
            else:
                ts = time.time()
                with SATSolver(name='g3') as s:
                    s.append_formula(f.clauses)
                    sat = s.solve()
                times.append(time.time() - ts)
                labels.append(1 if sat else 0)

        self._elapsed = time.time() - t0
        dist = Counter(labels)
        self.log(f"SAT={dist[1]}  UNSAT={dist[0]} — {self._elapsed:.2f}s")

        result = {
            'labels':       labels,
            'solve_times':  times,
            'distribution': dict(dist),
        }
        self.publish('A0', 'result', result)
        return result


# ══════════════════════════════════════════════════════════════════════
#  MODÈLE TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class _Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, drop: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop),
            nn.Linear(ff, d), nn.Dropout(drop),
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=mask, need_weights=False)
        x    = self.norm1(x + self.drop(a))
        return self.norm2(x + self.ff(x))


class SATTransformer(nn.Module):
    """
    Encoder-only Transformer pour classification binaire SAT / UNSAT.
    Token [CLS] en tête de séquence — sa représentation finale
    est passée à la tête de classification.
    """

    def __init__(self, vocab_size: int, pad_id: int,
                 d: int = 128, heads: int = 4,
                 n_layers: int = 3, ff: int = 512, drop: float = 0.1):
        super().__init__()
        self.pad_id    = pad_id
        self.embed     = nn.Embedding(vocab_size, d, padding_idx=pad_id)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d))
        self.pos_embed = nn.Parameter(torch.randn(1, MAX_LEN + 1, d))
        self.drop_e    = nn.Dropout(drop)
        self.blocks    = nn.ModuleList([_Block(d, heads, ff, drop) for _ in range(n_layers)])
        self.head      = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B    = x.size(0)
        mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=x.device),
            x == self.pad_id,
        ], dim=1)
        h = self.embed(x)
        h = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)
        h = self.drop_e(h + self.pos_embed[:, :h.size(1)])
        for block in self.blocks:
            h = block(h, mask)
        return self.head(h[:, 0]).squeeze(-1)


class _SATDataset(Dataset):
    def __init__(self, items: List[Tuple[torch.Tensor, int]]):
        self.items = items

    def __len__(self):  return len(self.items)

    def __getitem__(self, i):
        ids, lbl = self.items[i]
        return ids, torch.tensor(lbl, dtype=torch.float)


# ══════════════════════════════════════════════════════════════════════
#  AGENT A3 — TRANSFORMER NEURONAL
# ══════════════════════════════════════════════════════════════════════

class TransformerAgent(Agent):
    """
    Entraîne le SATTransformer sur GPU(s).
    DataParallel activé automatiquement si 2 T4 disponibles.
    Retourne probabilités, prédictions et historique d'entraînement.
    """

    def __init__(self, bus: MessageBus, vocab_size: int, pad_id: int):
        super().__init__('A3', bus)
        self.model = SATTransformer(vocab_size=vocab_size, pad_id=pad_id).to(DEVICE)
        if USE_DP:
            self.model = nn.DataParallel(self.model, device_ids=[0, 1])
            self.log("DataParallel sur GPU 0 + GPU 1")

    def _eval_loop(self, loader, loss_fn) -> Tuple[float, float]:
        self.model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits    = self.model(x)
                loss_sum += loss_fn(logits, y).item() * x.size(0)
                preds     = (torch.sigmoid(logits) > 0.5).float()
                correct  += (preds == y).sum().item()
                total    += y.size(0)
        return correct / total, loss_sum / total

    def run(self, train_data, val_data, test_data) -> Dict:
        t0 = time.time()
        self.log(f"train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

        g = torch.Generator(); g.manual_seed(SEED)
        mk = lambda data, shuffle: DataLoader(
            _SATDataset(data), batch_size=BATCH_SIZE,
            shuffle=shuffle, generator=(g if shuffle else None),
            pin_memory=True, num_workers=2,
        )
        train_loader = mk(train_data, True)
        val_loader   = mk(val_data,   False)
        test_loader  = mk(test_data,  False)

        loss_fn   = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        history = {k: [] for k in ('train_loss', 'val_loss', 'train_acc', 'val_acc')}
        best_score, best_state, patience_ctr = -float('inf'), None, 0

        print("\n" + "═" * 68)
        print("  A3 — ENTRAÎNEMENT")
        print("═" * 68)

        for epoch in range(EPOCHS):
            self.model.train()
            loss_sum, correct, total = 0.0, 0, 0
            t_ep = time.time()

            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits   = self.model(x)
                loss     = loss_fn(logits, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item() * x.size(0)
                preds     = (torch.sigmoid(logits) > 0.5).float()
                correct  += (preds == y).sum().item()
                total    += y.size(0)

            tr_loss = loss_sum / total
            tr_acc  = correct  / total
            vl_acc, vl_loss = self._eval_loop(val_loader, loss_fn)
            scheduler.step()

            history['train_loss'].append(tr_loss)
            history['val_loss'].append(vl_loss)
            history['train_acc'].append(tr_acc)
            history['val_acc'].append(vl_acc)

            score = vl_acc - vl_loss
            tag   = ''
            if score > best_score:
                best_score  = score
                best_state  = copy.deepcopy(self.model.state_dict())
                patience_ctr = 0
                tag = ' ★'
            else:
                patience_ctr += 1

            print(f"  Epoch {epoch+1:02d}/{EPOCHS}  "
                  f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
                  f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  "
                  f"{time.time()-t_ep:.1f}s{tag}")

            if patience_ctr >= PATIENCE:
                self.log(f"Early stopping — epoch {epoch+1}")
                break

            torch.cuda.empty_cache(); gc.collect()

        if best_state:
            self.model.load_state_dict(best_state)

        # Inférence sur le test set
        self.model.eval()
        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE)
                p = torch.sigmoid(self.model(x)).cpu().numpy()
                all_probs.extend(p.tolist())
                all_preds.extend((p > 0.5).astype(int).tolist())
                all_labels.extend(y.numpy().astype(int).tolist())

        self._elapsed = time.time() - t0
        result = {
            'probs':   all_probs,
            'preds':   all_preds,
            'labels':  all_labels,
            'history': history,
            'model':   self.model,
        }
        self.publish('A0', 'result', result)
        return result


# ══════════════════════════════════════════════════════════════════════
#  AGENT A4 — ÉVALUATEUR
# ══════════════════════════════════════════════════════════════════════

class EvaluatorAgent(Agent):
    """
    Calcule les métriques de classification et génère le rapport graphique
    complet (10 panneaux) sauvegardé dans mas_sat_report.png.
    """

    # ── Palette ────────────────────────────────────────────────────────
    C = {
        'bg':    '#0D0D1A', 'panel': '#13132B',
        'c1':    '#00D4FF', 'c2':    '#FF6B6B',
        'c3':    '#A8FF78', 'c4':    '#FFD93D',
        'c5':    '#C77DFF', 'text':  '#DDE0FF',
        'grid':  '#252545',
    }

    def _apply_style(self):
        C = self.C
        plt.rcParams.update({
            'figure.facecolor': C['bg'],   'axes.facecolor':  C['panel'],
            'axes.edgecolor':   C['grid'], 'axes.labelcolor': C['text'],
            'xtick.color':      C['text'], 'ytick.color':     C['text'],
            'text.color':       C['text'], 'grid.color':      C['grid'],
            'grid.linewidth':   0.5,       'font.family':     'monospace',
            'legend.facecolor': C['panel'],'legend.edgecolor': C['grid'],
        })

    def run(self, trans: Dict, sym: Dict, prep: Dict,
            timings: Dict, bus_stats: Dict) -> Dict:
        t0 = time.time()
        self.log("Calcul des métriques")

        probs   = np.array(trans['probs'])
        preds   = np.array(trans['preds'])
        labels  = np.array(trans['labels'])
        history = trans['history']

        acc  = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec  = recall_score(labels, preds, zero_division=0)
        f1   = f1_score(labels, preds, zero_division=0)
        auc  = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float('nan')
        cm   = confusion_matrix(labels, preds, labels=[0, 1])

        metrics = dict(accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc)

        print("\n" + "═" * 68)
        print("  A4 — MÉTRIQUES FINALES")
        print("═" * 68)
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-score  : {f1:.4f}")
        print(f"  ROC-AUC   : {auc:.4f}")
        print()
        print(classification_report(
            labels, preds,
            target_names=['UNSAT', 'SAT'],
            digits=4, zero_division=0,
        ))

        self._plot(history, probs, labels, cm, metrics,
                   timings, bus_stats, sym, prep)

        self._elapsed = time.time() - t0
        self.log(f"Rapport sauvegardé — {self._elapsed:.2f}s")
        self.publish('A0', 'result', metrics)
        return metrics

    def _plot(self, history, probs, labels, cm, metrics,
              timings, bus_stats, sym, prep):
        self._apply_style()
        C  = self.C
        ep = range(1, len(history['train_loss']) + 1)

        fig = plt.figure(figsize=(22, 26), facecolor=C['bg'])
        fig.suptitle(
            "MAS-SAT  ·  Rapport Technique  ·  Système Multi-Agent",
            fontsize=17, color=C['c1'], fontweight='bold', y=0.995,
        )
        gs = gridspec.GridSpec(
            4, 3, figure=fig,
            hspace=0.48, wspace=0.35,
            top=0.965, bottom=0.04, left=0.06, right=0.97,
        )

        # 1 — Loss
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(ep, history['train_loss'], color=C['c1'], lw=2, label='Train')
        ax.plot(ep, history['val_loss'],   color=C['c2'], lw=2, ls='--', label='Val')
        ax.set_title("Loss (BCE)", color=C['c1'], fontweight='bold')
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True)

        # 2 — Accuracy
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(ep, history['train_acc'], color=C['c3'], lw=2, label='Train')
        ax.plot(ep, history['val_acc'],   color=C['c4'], lw=2, ls='--', label='Val')
        ax.set_title("Accuracy", color=C['c3'], fontweight='bold')
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1); ax.legend(); ax.grid(True)

        # 3 — Distribution P(SAT)
        ax = fig.add_subplot(gs[0, 2])
        ax.hist(probs[labels == 1], bins=30, alpha=0.75, color=C['c3'], label='SAT')
        ax.hist(probs[labels == 0], bins=30, alpha=0.75, color=C['c2'], label='UNSAT')
        ax.axvline(0.5, color='white', ls='--', lw=1.5, label='seuil')
        ax.set_title("Distribution P(SAT)", color=C['c4'], fontweight='bold')
        ax.set_xlabel("P(SAT)"); ax.set_ylabel("Fréquence")
        ax.legend(); ax.grid(True)

        # 4 — Matrice de confusion
        ax = fig.add_subplot(gs[1, 0])
        im = ax.imshow(cm, cmap='Blues')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=18, fontweight='bold',
                        color='white' if cm[i, j] > cm.max() / 2 else C['text'])
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['UNSAT', 'SAT'])
        ax.set_yticklabels(['UNSAT', 'SAT'])
        ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
        ax.set_title("Matrice de Confusion", color=C['c2'], fontweight='bold')

        # 5 — Radar métriques
        ax = fig.add_subplot(gs[1, 1], polar=True)
        cats   = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        vals   = [metrics[k] for k in ('accuracy', 'precision', 'recall', 'f1', 'auc')]
        N      = len(cats)
        angles = [n / N * 2 * np.pi for n in range(N)] + [0]
        vals_r = vals + vals[:1]
        ax.plot(angles, vals_r, color=C['c1'], lw=2)
        ax.fill(angles, vals_r, color=C['c1'], alpha=0.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cats, color=C['text'], size=9)
        ax.set_ylim(0, 1)
        ax.set_facecolor(C['panel'])
        ax.spines['polar'].set_color(C['grid'])
        ax.yaxis.set_tick_params(labelcolor=C['grid'])
        ax.set_title("Radar Métriques", color=C['c1'], fontweight='bold', pad=15)

        # 6 — Temps par agent
        ax = fig.add_subplot(gs[1, 2])
        names  = list(timings.keys())
        values = list(timings.values())
        colors = [C['c1'], C['c2'], C['c3'], C['c4'], C['c5']][:len(names)]
        bars   = ax.barh(names, values, color=colors, edgecolor='none', height=0.55)
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{v:.1f}s', va='center', ha='left', fontsize=9)
        ax.set_title("Temps par Agent", color=C['c4'], fontweight='bold')
        ax.set_xlabel("Secondes"); ax.grid(True, axis='x')

        # 7 — Messages inter-agents
        ax = fig.add_subplot(gs[2, 0])
        ch  = bus_stats['channels']
        pal = plt.cm.plasma(np.linspace(0.2, 0.9, len(ch)))
        wedges, _, autotexts = ax.pie(
            list(ch.values()), labels=list(ch.keys()),
            autopct='%1.0f%%', colors=pal,
            textprops={'fontsize': 7, 'color': C['text']},
            startangle=140,
        )
        for at in autotexts:
            at.set_color('white'); at.set_fontsize(8)
        ax.set_title("Messages Inter-Agents", color=C['c3'], fontweight='bold')

        # 8 — Ratio clauses/variables
        ax = fig.add_subplot(gs[2, 1])
        ratios = [s['ratio'] for s in prep.get('stats_list', [])]
        if ratios:
            ax.hist(ratios, bins=30, color=C['c4'], edgecolor='none', alpha=0.85)
            ax.axvline(np.mean(ratios), color=C['c2'], ls='--', lw=2,
                       label=f'μ = {np.mean(ratios):.2f}')
            ax.set_title("Ratio Clauses / Variables", color=C['c4'], fontweight='bold')
            ax.set_xlabel("n_clauses / n_vars"); ax.set_ylabel("Fréquence")
            ax.legend(); ax.grid(True)

        # 9 — Temps résolution PySAT
        ax = fig.add_subplot(gs[2, 2])
        st = np.array(sym.get('solve_times', []))
        if st.max() > 0:
            ax.hist(st * 1000, bins=30, color=C['c1'], edgecolor='none', alpha=0.85)
            ax.set_title("Temps Résolution PySAT (ms)", color=C['c1'], fontweight='bold')
            ax.set_xlabel("ms"); ax.set_ylabel("Fréquence"); ax.grid(True)
        else:
            ax.text(0.5, 0.5, "Labels pré-calculés\n(synthétique)",
                    ha='center', va='center', transform=ax.transAxes,
                    color=C['text'], fontsize=11)
            ax.set_title("Temps Résolution PySAT", color=C['c1'], fontweight='bold')

        # 10 — Diagramme d'architecture MAS
        ax = fig.add_subplot(gs[3, :])
        ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis('off')
        ax.set_title("Architecture du Système Multi-Agent",
                     color=C['c1'], fontweight='bold', pad=10)

        agents = [
            (0.9,  1.5, "A0\nOrchestrator",       C['c4']),
            (3.1,  2.3, "A1\nPréprocesseur",       C['c1']),
            (3.1,  0.7, "A2\nSolveur exact",       C['c2']),
            (6.0,  1.5, "A3\nTransformer (GPU)",   C['c3']),
            (8.9,  1.5, "A4\nÉvaluateur",          C['c5']),
        ]
        BW, BH = 1.5, 0.85
        for (x, y, lbl, col) in agents:
            rect = mpatches.FancyBboxPatch(
                (x - BW/2, y - BH/2), BW, BH,
                boxstyle='round,pad=0.07',
                facecolor=col + '28', edgecolor=col, linewidth=2.5,
            )
            ax.add_patch(rect)
            ax.text(x, y, lbl, ha='center', va='center',
                    color='white', fontsize=8.5, fontweight='bold', linespacing=1.5)

        for (x1, y1, x2, y2) in [
            (1.65, 1.5,  2.35, 2.3),
            (1.65, 1.5,  2.35, 0.7),
            (3.85, 2.3,  5.25, 1.7),
            (3.85, 0.7,  5.25, 1.3),
            (6.75, 1.5,  8.15, 1.5),
        ]:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(
                            arrowstyle='->', color=C['grid'],
                            lw=2, connectionstyle='arc3,rad=0.1',
                        ))

        ax.text(5.0, 0.18,
                "── MessageBus  (publish / consume) ──",
                ha='center', va='center', fontsize=9,
                color=C['grid'], style='italic')

        summary = (
            f"Accuracy={metrics['accuracy']:.3f}   "
            f"Precision={metrics['precision']:.3f}   "
            f"Recall={metrics['recall']:.3f}   "
            f"F1={metrics['f1']:.3f}   "
            f"AUC={metrics['auc']:.3f}   "
            f"│  Messages={bus_stats['total']}"
        )
        ax.text(5.0, 2.82, summary, ha='center', va='center', fontsize=9,
                color=C['c4'],
                bbox=dict(facecolor=C['panel'], edgecolor=C['c4'],
                          boxstyle='round,pad=0.4', lw=1.5))

        plt.savefig('mas_sat_report.png', dpi=130,
                    bbox_inches='tight', facecolor=C['bg'])
        plt.close(fig)
        print("  ✓ mas_sat_report.png sauvegardé")


# ══════════════════════════════════════════════════════════════════════
#  AGENT A0 — ORCHESTRATEUR
# ══════════════════════════════════════════════════════════════════════

class OrchestratorAgent(Agent):
    """
    Coordonne l'ensemble du pipeline MAS dans cet ordre :
      données → A1 → A2 → split → A3 → A4 → sauvegarde
    """

    def run(
        self,
        satlib_dir:  str = '/kaggle/working/generated_3sat',
        n_instances: int = 2000,
        n_vars:      int = 20,
        n_clauses:   int = 85,
    ) -> Dict:
        t_start = time.time()

        print("\n╔" + "═" * 66 + "╗")
        print("║" + "  MAS-SAT — PIPELINE MULTI-AGENT".center(66) + "║")
        print("╚" + "═" * 66 + "╝\n")

        # ── Données (SATLIB — fallback synthétique si nécessaire) ──
        self.log("Chargement SATLIB")
        try:
            instances = load_satlib(satlib_dir, max_instances=n_instances)
            dist = Counter()
            for f in instances:
                with SATSolver(name='g3') as s:
                    s.append_formula(f.clauses)
                    dist[s.solve()] += 1
            if dist[True] == 0 or dist[False] == 0:
                raise ValueError("Dataset déséquilibré")
        except Exception as e:
            self.log(f"SATLIB indisponible ({e}) — fallback synthétique")
            instances = generate_fallback(
                n_vars, n_clauses,
                n_instances // 2, n_instances // 2,
            )

        # ── A1 — Préprocesseur ─────────────────────────────────────
        self.log("Lancement A1 — Préprocesseur")
        a1         = PreprocessorAgent('A1', self.bus)
        prep       = a1.run(instances)

        # ── A2 — Solveur exact ─────────────────────────────────────
        self.log("Lancement A2 — Solveur exact")
        a2         = SymbolicSolverAgent('A2', self.bus)
        sym        = a2.run(instances)

        # ── Split train / val / test ───────────────────────────────
        self.log("Split 80 / 10 / 10")
        items = list(zip(prep['tokenized'], sym['labels']))
        idx   = torch.randperm(len(items),
                               generator=torch.Generator().manual_seed(SEED))
        n      = len(items)
        n_tr   = int(0.80 * n)
        n_vl   = int(0.10 * n)
        train  = [items[i] for i in idx[:n_tr]]
        val    = [items[i] for i in idx[n_tr: n_tr + n_vl]]
        test   = [items[i] for i in idx[n_tr + n_vl:]]

        # ── A3 — Transformer ───────────────────────────────────────
        self.log("Lancement A3 — Transformer")
        a3    = TransformerAgent(self.bus,
                                 vocab_size=prep['vocab_size'],
                                 pad_id=prep['token2id']['PAD'])
        trans = a3.run(train, val, test)

        # ── A4 — Évaluateur ────────────────────────────────────────
        self.log("Lancement A4 — Évaluateur")
        timings = {
            'A1 — Préprocesseur':  a1.elapsed,
            'A2 — Solveur exact':  a2.elapsed,
            'A3 — Transformer':    a3.elapsed,
        }
        a4      = EvaluatorAgent('A4', self.bus)
        metrics = a4.run(trans, sym, prep, timings, self.bus.stats)
        timings['A4 — Évaluateur'] = a4.elapsed

        # ── Sauvegarde du modèle ───────────────────────────────────
        core = (trans['model'].module
                if isinstance(trans['model'], nn.DataParallel)
                else trans['model'])
        torch.save({
            'model_state_dict': core.state_dict(),
            'token2id':         prep['token2id'],
            'vocab_size':       prep['vocab_size'],
            'config': dict(d=128, heads=4, n_layers=3, ff=512,
                           max_len=MAX_LEN, drop=0.1),
            'metrics':          metrics,
        }, 'mas_sat_model.pt')

        # ── Rapport final ──────────────────────────────────────────
        t_total = time.time() - t_start
        print("\n╔" + "═" * 66 + "╗")
        print("║" + "  RÉSULTATS".center(66) + "║")
        print("╠" + "═" * 66 + "╣")
        print(f"║  Instances        : {len(instances):<47}║")
        print(f"║  Vocab size       : {prep['vocab_size']:<47}║")
        print(f"║  Distribution     : {sym['distribution']!s:<47}║")
        print(f"║  GPU(s)           : {_n_gpu:<47}║")
        print("╠" + "═" * 66 + "╣")
        for k, v in metrics.items():
            print(f"║  {k:<17}: {v:.4f}{' '*42}║")
        print("╠" + "═" * 66 + "╣")
        print(f"║  Temps total      : {t_total:.1f}s{' '*43}║")
        print(f"║  Messages échangés: {self.bus.stats['total']:<47}║")
        print("╚" + "═" * 66 + "╝")
        print("\n  ✓ mas_sat_model.pt  ✓ mas_sat_report.png\n")

        return metrics


# ══════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    bus          = MessageBus()
    orchestrator = OrchestratorAgent('A0', bus)

    orchestrator.run(
        satlib_dir  = '/kaggle/working/generated_3sat',
        n_instances = 2000,
        n_vars      = 20,      # utilisé uniquement en fallback synthétique
        n_clauses   = 85,      # ratio ≈ 4.25 → zone de transition de phase
    )