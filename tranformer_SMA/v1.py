"""version 1"""

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         MAS-SAT : Système Multi-Agent pour la résolution SAT                ║
║         Architecture : Orchestrateur + Préprocesseur + Solveur              ║
║                        Symbolique + Transformer + Évaluateur                ║
║         Environnement : Kaggle — 2x NVIDIA Tesla T4                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Agents :
  [A0] OrchestratorAgent   — Pipeline global, distribution des tâches
  [A1] PreprocessorAgent   — Analyse et tokenisation des formules CNF
  [A2] SymbolicSolverAgent — Résolution exacte via PySAT (ground truth)
  [A3] TransformerAgent    — Classification neuronale (GPU)
  [A4] EvaluatorAgent      — Métriques, rapports et graphiques

Communication : Bus de messages (dict Python) avec protocole simple
"""

# ─────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────
import os, time, copy, random, gc, warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from pysat.formula import CNF
from pysat.solvers import Solver as SATSolver
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION GLOBALE
# ─────────────────────────────────────────────────────────────
SEED        = 42
MAX_LEN     = 512
EPOCHS      = 25
BATCH_SIZE  = 16
PATIENCE    = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# Multi-GPU
if torch.cuda.device_count() >= 2:
    DEVICE_PRIMARY   = torch.device('cuda:0')
    DEVICE_SECONDARY = torch.device('cuda:1')
    print(f"[CONFIG] Deux GPU détectés : {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_name(1)}")
elif torch.cuda.is_available():
    DEVICE_PRIMARY   = torch.device('cuda:0')
    DEVICE_SECONDARY = torch.device('cuda:0')
    print(f"[CONFIG] Un GPU détecté : {torch.cuda.get_device_name(0)}")
else:
    DEVICE_PRIMARY   = torch.device('cpu')
    DEVICE_SECONDARY = torch.device('cpu')
    print("[CONFIG] CPU mode")


# ═════════════════════════════════════════════════════════════
#  SECTION 1 — BUS DE COMMUNICATION INTER-AGENTS
# ═════════════════════════════════════════════════════════════

@dataclass
class Message:
    """Enveloppe de communication entre agents."""
    sender:    str
    receiver:  str
    msg_type:  str           # 'task' | 'result' | 'status' | 'error'
    payload:   Any = None
    timestamp: float = field(default_factory=time.time)

    def __repr__(self):
        return (f"[{self.msg_type.upper()}] {self.sender} → {self.receiver} "
                f"| {str(self.payload)[:60]}...")


class MessageBus:
    """
    Bus de messages synchrone.
    Chaque agent dépose ses messages ici ; les autres les consomment.
    Simule un middleware léger (type publish-subscribe simplifié).
    """

    def __init__(self):
        self._queues:   Dict[str, List[Message]] = defaultdict(list)
        self._log:      List[Message]            = []
        self._counters: Dict[str, int]           = defaultdict(int)

    def send(self, msg: Message):
        self._queues[msg.receiver].append(msg)
        self._log.append(msg)
        self._counters[f"{msg.sender}→{msg.receiver}"] += 1

    def receive(self, agent_id: str) -> List[Message]:
        msgs = self._queues.pop(agent_id, [])
        return msgs

    def stats(self) -> Dict:
        return {
            'total_messages': len(self._log),
            'channels':       dict(self._counters)
        }


# ═════════════════════════════════════════════════════════════
#  SECTION 2 — CLASSE DE BASE AGENT
# ═════════════════════════════════════════════════════════════

class BaseAgent(ABC):
    """
    Contrat commun à tous les agents.
    Chaque agent possède :
      - un identifiant unique
      - un accès au bus de communication
      - un journal d'activité interne
      - un chronomètre cumulé
    """

    def __init__(self, agent_id: str, bus: MessageBus):
        self.agent_id  = agent_id
        self.bus       = bus
        self._log      = []
        self._t_total  = 0.0

    def log(self, msg: str):
        entry = f"[{self.agent_id}] {msg}"
        self._log.append(entry)
        print(entry)

    def send(self, receiver: str, msg_type: str, payload: Any = None):
        self.bus.send(Message(
            sender=self.agent_id,
            receiver=receiver,
            msg_type=msg_type,
            payload=payload
        ))

    def receive(self) -> List[Message]:
        return self.bus.receive(self.agent_id)

    @abstractmethod
    def run(self, *args, **kwargs):
        """Point d'entrée principal de l'agent."""
        ...

    @property
    def elapsed(self) -> float:
        return self._t_total


# ═════════════════════════════════════════════════════════════
#  SECTION 3 — GÉNÉRATION DE DONNÉES SAT
# ═════════════════════════════════════════════════════════════

def generate_balanced_dataset(
    n_vars: int, n_clauses: int,
    n_sat: int, n_unsat: int, seed: int
) -> List[CNF]:
    """Génère des instances 3-SAT synthétiques équilibrées via PySAT."""
    random.seed(seed); np.random.seed(seed)

    def random_clause(nv, k=3):
        vs = random.sample(range(1, nv + 1), k)
        return [v * random.choice([-1, 1]) for v in vs]

    instances, sat_c, unsat_c, attempts = [], 0, 0, 0
    max_att = (n_sat + n_unsat) * 30

    while (sat_c < n_sat or unsat_c < n_unsat) and attempts < max_att:
        attempts += 1
        clauses = [random_clause(n_vars) for _ in range(n_clauses)]
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

    print(f"  ✓ SAT={sat_c}  UNSAT={unsat_c}  total={len(instances)}")
    return instances


def load_satlib(directory: str, max_instances: int) -> List[CNF]:
    """Charge des fichiers .cnf depuis un répertoire SATLIB."""
    files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory) if f.endswith('.cnf')
    ])[:max_instances]
    instances = []
    for path in files:
        try:
            f = CNF(from_file=path)
            f.path = path
            instances.append(f)
        except Exception as e:
            print(f"  ✗ Erreur {path}: {e}")
    print(f"  ✓ {len(instances)} instances SATLIB chargées")
    return instances


# ═════════════════════════════════════════════════════════════
#  SECTION 4 — AGENT A1 : PRÉPROCESSEUR
# ═════════════════════════════════════════════════════════════

SPECIAL_TOKENS = ['PAD', 'AND', 'OR', 'NOT', 'CLS']

class PreprocessorAgent(BaseAgent):
    """
    Agent A1 — Préprocesseur
    ────────────────────────
    Responsabilités :
      • Construire le vocabulaire à partir des formules CNF
      • Convertir chaque formule en séquence de tokens (ids)
      • Extraire des statistiques structurelles par formule
      • Publier les données tokenisées sur le bus
    """

    def __init__(self, bus: MessageBus):
        super().__init__("A1-Preprocessor", bus)
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.formula_stats: List[Dict] = []

    def _build_vocab(self, instances: List[CNF]):
        vars_set = set()
        for f in instances:
            for clause in f.clauses:
                for lit in clause:
                    vars_set.add(abs(lit))
        self.token2id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        for v in sorted(vars_set):
            self.token2id[f'x{v}'] = len(self.token2id)
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.log(f"Vocabulaire construit : {len(self.token2id)} tokens")

    def _formula_to_ids(self, formula: CNF) -> torch.Tensor:
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
        ids = [self.token2id.get(t, self.token2id['PAD']) for t in tokens]
        ids = ids[:MAX_LEN]
        ids += [self.token2id['PAD']] * (MAX_LEN - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def _extract_stats(self, formula: CNF) -> Dict:
        """Statistiques structurelles d'une formule."""
        n_clauses = len(formula.clauses)
        n_vars    = formula.nv
        lits_per_clause = [len(c) for c in formula.clauses]
        neg_ratio = sum(1 for c in formula.clauses for l in c if l < 0) / \
                    max(1, sum(len(c) for c in formula.clauses))
        return {
            'n_clauses':    n_clauses,
            'n_vars':       n_vars,
            'ratio':        n_clauses / max(1, n_vars),   # ratio clause/variable
            'avg_lit':      np.mean(lits_per_clause),
            'neg_ratio':    neg_ratio,
        }

    def run(self, instances: List[CNF]) -> Dict:
        t0 = time.time()
        self.log(f"Démarrage — {len(instances)} formules à traiter")

        self._build_vocab(instances)

        tokenized, stats_list = [], []
        for f in instances:
            ids   = self._formula_to_ids(f)
            stats = self._extract_stats(f)
            tokenized.append(ids)
            stats_list.append(stats)

        self._t_total = time.time() - t0
        self.log(f"Terminé en {self._t_total:.2f}s")

        result = {
            'token2id':    self.token2id,
            'id2token':    self.id2token,
            'vocab_size':  len(self.token2id),
            'tokenized':   tokenized,       # List[Tensor]
            'stats_list':  stats_list,      # List[Dict]
        }
        self.send("A0-Orchestrator", "result", result)
        return result


# ═════════════════════════════════════════════════════════════
#  SECTION 5 — AGENT A2 : SOLVEUR SYMBOLIQUE
# ═════════════════════════════════════════════════════════════

class SymbolicSolverAgent(BaseAgent):
    """
    Agent A2 — Solveur Symbolique (PySAT)
    ──────────────────────────────────────
    Responsabilités :
      • Résoudre chaque formule CNF de manière exacte (DPLL / Glucose3)
      • Fournir les labels ground truth (SAT=1, UNSAT=0)
      • Collecter des métriques de résolution (temps, satisfaisabilité)
    """

    def run(self, instances: List[CNF]) -> Dict:
        t0 = time.time()
        self.log(f"Résolution de {len(instances)} instances via PySAT (g3)")
        labels, times, solve_stats = [], [], []

        for f in instances:
            if hasattr(f, '_label'):
                label = f._label
                solve_time = 0.0
            else:
                t_s = time.time()
                with SATSolver(name='g3') as s:
                    s.append_formula(f.clauses)
                    is_sat = s.solve()
                solve_time = time.time() - t_s
                label = 1 if is_sat else 0

            labels.append(label)
            times.append(solve_time)
            solve_stats.append({'label': label, 'solve_time': solve_time})

        self._t_total = time.time() - t0
        dist = Counter(labels)
        self.log(f"Résolution terminée en {self._t_total:.2f}s | SAT={dist[1]}  UNSAT={dist[0]}")

        result = {
            'labels':       labels,
            'solve_times':  times,
            'solve_stats':  solve_stats,
            'distribution': dict(dist),
        }
        self.send("A0-Orchestrator", "result", result)
        return result


# ═════════════════════════════════════════════════════════════
#  SECTION 6 — MODÈLE TRANSFORMER (pour Agent A3)
# ═════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim), nn.Dropout(dropout)
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=mask, need_weights=False)
        x    = self.norm1(x + self.drop(a))
        return self.norm2(x + self.ff(x))


class SATTransformer(nn.Module):
    """
    Transformer encoder pour classification SAT/UNSAT.
    Inspiré de BERT : token [CLS] + position embeddings apprenables.
    """

    def __init__(self, vocab_size, pad_id, hidden_dim=128,
                 n_heads=4, n_layers=3, ff_dim=512, dropout=0.1):
        super().__init__()
        self.pad_id    = pad_id
        self.embed     = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, MAX_LEN + 1, hidden_dim))
        self.drop_e    = nn.Dropout(dropout)
        self.layers    = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x_ids):
        B = x_ids.size(0)
        mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=x_ids.device),
            (x_ids == self.pad_id)
        ], dim=1)
        x = self.embed(x_ids)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.drop_e(x + self.pos_embed[:, :x.size(1)])
        for layer in self.layers:
            x = layer(x, mask)
        return self.head(x[:, 0]).squeeze(-1)


class SATDataset(Dataset):
    def __init__(self, items):       # items : List[(tensor_ids, label)]
        self.items = items
    def __len__(self):  return len(self.items)
    def __getitem__(self, i):
        ids, lbl = self.items[i]
        return ids, torch.tensor(lbl, dtype=torch.float)


# ═════════════════════════════════════════════════════════════
#  SECTION 7 — AGENT A3 : TRANSFORMER
# ═════════════════════════════════════════════════════════════

class TransformerAgent(BaseAgent):
    """
    Agent A3 — Transformer Neuronal
    ────────────────────────────────
    Responsabilités :
      • Instancier et entraîner le SATTransformer sur GPU(s)
      • Utiliser DataParallel si 2 GPU disponibles
      • Early stopping + scheduler
      • Retourner probabilités et prédictions sur le test set
    """

    def __init__(self, bus: MessageBus, vocab_size: int, pad_id: int):
        super().__init__("A3-Transformer", bus)
        self.model = SATTransformer(
            vocab_size=vocab_size,
            pad_id=pad_id,
            hidden_dim=128,
            n_heads=4,
            n_layers=3,
            ff_dim=512,
            dropout=0.1
        ).to(DEVICE_PRIMARY)

        # Multi-GPU DataParallel
        if torch.cuda.device_count() >= 2:
            self.model = nn.DataParallel(self.model, device_ids=[0, 1])
            self.log("DataParallel activé sur GPU 0 et GPU 1")

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc':  [], 'val_acc':  []
        }

    def _evaluate(self, loader, loss_fn):
        self.model.eval()
        tot_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE_PRIMARY), y.to(DEVICE_PRIMARY)
                logits = self.model(x)
                tot_loss += loss_fn(logits, y).item() * x.size(0)
                preds     = (torch.sigmoid(logits) > 0.5).float()
                correct  += (preds == y).sum().item()
                total    += y.size(0)
        return correct / total, tot_loss / total

    def run(self, train_data, val_data, test_data) -> Dict:
        t0 = time.time()
        self.log(f"Entraînement : train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

        g = torch.Generator(); g.manual_seed(SEED)
        train_loader = DataLoader(SATDataset(train_data), batch_size=BATCH_SIZE,
                                  shuffle=True, generator=g, pin_memory=True)
        val_loader   = DataLoader(SATDataset(val_data),   batch_size=BATCH_SIZE, pin_memory=True)
        test_loader  = DataLoader(SATDataset(test_data),  batch_size=BATCH_SIZE, pin_memory=True)

        loss_fn   = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_score, best_state, counter = -float('inf'), None, 0

        print("\n" + "═" * 72)
        print("  AGENT A3 — TRANSFORMER TRAINING")
        print("═" * 72)

        for epoch in range(EPOCHS):
            self.model.train()
            tot_loss, correct, total = 0, 0, 0
            t_ep = time.time()

            for x, y in train_loader:
                x, y = x.to(DEVICE_PRIMARY), y.to(DEVICE_PRIMARY)
                logits = self.model(x)
                loss   = loss_fn(logits, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                tot_loss += loss.item() * x.size(0)
                preds     = (torch.sigmoid(logits) > 0.5).float()
                correct  += (preds == y).sum().item()
                total    += y.size(0)

            tr_loss = tot_loss / total
            tr_acc  = correct  / total
            vl_acc, vl_loss = self._evaluate(val_loader, loss_fn)
            scheduler.step()

            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(vl_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_acc'].append(vl_acc)

            score = vl_acc - vl_loss
            tag   = ""
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(self.model.state_dict())
                counter    = 0
                tag        = " ★ best"
            else:
                counter += 1

            print(f"  Epoch {epoch+1:02d}/{EPOCHS} | "
                  f"TrLoss={tr_loss:.4f} TrAcc={tr_acc:.4f} | "
                  f"ValLoss={vl_loss:.4f} ValAcc={vl_acc:.4f} | "
                  f"{time.time()-t_ep:.1f}s{tag}")

            if counter >= PATIENCE:
                self.log(f"Early stopping à l'epoch {epoch+1}")
                break

        # Charger le meilleur modèle
        if best_state:
            self.model.load_state_dict(best_state)

        # Prédictions sur le test set
        self.model.eval()
        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE_PRIMARY)
                probs = torch.sigmoid(self.model(x)).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_labels.extend(y.numpy().astype(int).tolist())

        self._t_total = time.time() - t0
        self.log(f"Entraînement terminé en {self._t_total:.2f}s")

        result = {
            'probs':   all_probs,
            'preds':   all_preds,
            'labels':  all_labels,
            'history': self.history,
        }
        self.send("A0-Orchestrator", "result", result)
        return result


# ═════════════════════════════════════════════════════════════
#  SECTION 8 — AGENT A4 : ÉVALUATEUR
# ═════════════════════════════════════════════════════════════

class EvaluatorAgent(BaseAgent):
    """
    Agent A4 — Évaluateur & Rapporteur
    ────────────────────────────────────
    Responsabilités :
      • Calculer toutes les métriques de classification
      • Comparer solveur symbolique vs. transformer
      • Générer les graphiques du rapport technique
      • Sauvegarder les figures
    """

    def run(
        self,
        transformer_result: Dict,
        symbolic_result:    Dict,
        preprocess_result:  Dict,
        agent_timings:      Dict,
        bus_stats:          Dict,
    ) -> Dict:
        t0 = time.time()
        self.log("Calcul des métriques et génération des graphiques")

        probs  = np.array(transformer_result['probs'])
        preds  = np.array(transformer_result['preds'])
        labels = np.array(transformer_result['labels'])
        history = transformer_result['history']

        acc  = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec  = recall_score(labels, preds, zero_division=0)
        f1   = f1_score(labels, preds, zero_division=0)
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = float('nan')
        cm   = confusion_matrix(labels, preds, labels=[0, 1])

        metrics = dict(accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc)

        print("\n" + "═" * 72)
        print("  AGENT A4 — RAPPORT FINAL")
        print("═" * 72)
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-score  : {f1:.4f}")
        print(f"  ROC-AUC   : {auc:.4f}")
        print()
        print(classification_report(
            labels, preds,
            target_names=['UNSAT', 'SAT'],
            digits=4, zero_division=0
        ))

        # ── Graphiques ────────────────────────────────────────────
        self._plot_all(
            history=history,
            probs=probs, labels=labels,
            cm=cm,
            metrics=metrics,
            agent_timings=agent_timings,
            bus_stats=bus_stats,
            symbolic_result=symbolic_result,
            preprocess_result=preprocess_result,
        )

        self._t_total = time.time() - t0
        self.log(f"Rapport généré en {self._t_total:.2f}s")
        self.send("A0-Orchestrator", "result", metrics)
        return metrics

    # ── Helpers graphiques ────────────────────────────────────────

    def _plot_all(self, history, probs, labels, cm, metrics,
                  agent_timings, bus_stats, symbolic_result, preprocess_result):

        STYLE = {
            'bg':      '#0F0F1A',
            'panel':   '#1A1A2E',
            'accent1': '#00D4FF',
            'accent2': '#FF6B6B',
            'accent3': '#A8FF78',
            'accent4': '#FFD93D',
            'text':    '#E0E0FF',
            'grid':    '#2A2A4A',
        }
        plt.rcParams.update({
            'figure.facecolor': STYLE['bg'],
            'axes.facecolor':   STYLE['panel'],
            'axes.edgecolor':   STYLE['grid'],
            'axes.labelcolor':  STYLE['text'],
            'xtick.color':      STYLE['text'],
            'ytick.color':      STYLE['text'],
            'text.color':       STYLE['text'],
            'grid.color':       STYLE['grid'],
            'grid.linewidth':   0.5,
            'font.family':      'monospace',
        })

        fig = plt.figure(figsize=(22, 28), facecolor=STYLE['bg'])
        fig.suptitle(
            "MAS-SAT  ·  Rapport Technique  ·  Système Multi-Agent",
            fontsize=18, color=STYLE['accent1'], fontweight='bold', y=0.99
        )

        gs = gridspec.GridSpec(
            4, 3,
            figure=fig,
            hspace=0.45, wspace=0.35,
            top=0.96, bottom=0.04,
            left=0.06, right=0.97
        )

        # ── 1. Courbes d'entraînement — Loss ──────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ep  = range(1, len(history['train_loss']) + 1)
        ax1.plot(ep, history['train_loss'], color=STYLE['accent1'], lw=2, label='Train')
        ax1.plot(ep, history['val_loss'],   color=STYLE['accent2'], lw=2, label='Val', ls='--')
        ax1.set_title("Training Loss", color=STYLE['accent1'], fontweight='bold')
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
        ax1.legend(); ax1.grid(True)

        # ── 2. Courbes d'entraînement — Accuracy ──────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(ep, history['train_acc'], color=STYLE['accent3'], lw=2, label='Train')
        ax2.plot(ep, history['val_acc'],   color=STYLE['accent4'], lw=2, label='Val', ls='--')
        ax2.set_title("Training Accuracy", color=STYLE['accent3'], fontweight='bold')
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(True)

        # ── 3. Distribution des probabilités ──────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        sat_probs   = probs[labels == 1]
        unsat_probs = probs[labels == 0]
        ax3.hist(sat_probs,   bins=30, alpha=0.7, color=STYLE['accent3'], label='SAT')
        ax3.hist(unsat_probs, bins=30, alpha=0.7, color=STYLE['accent2'], label='UNSAT')
        ax3.axvline(0.5, color='white', ls='--', lw=1.5, label='seuil 0.5')
        ax3.set_title("Distribution des Probabilités", color=STYLE['accent4'], fontweight='bold')
        ax3.set_xlabel("P(SAT)"); ax3.set_ylabel("Fréquence")
        ax3.legend(); ax3.grid(True)

        # ── 4. Matrice de confusion ────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 0])
        im  = ax4.imshow(cm, cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        for i in range(2):
            for j in range(2):
                ax4.text(j, i, str(cm[i, j]), ha='center', va='center',
                         fontsize=18, fontweight='bold',
                         color='white' if cm[i, j] > cm.max() / 2 else STYLE['text'])
        ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
        ax4.set_xticklabels(['UNSAT', 'SAT']); ax4.set_yticklabels(['UNSAT', 'SAT'])
        ax4.set_xlabel("Prédit"); ax4.set_ylabel("Réel")
        ax4.set_title("Matrice de Confusion", color=STYLE['accent2'], fontweight='bold')

        # ── 5. Radar des métriques ─────────────────────────────────
        ax5 = fig.add_subplot(gs[1, 1], polar=True)
        cats   = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        vals   = [metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']]
        N      = len(cats)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]; vals_r = vals + vals[:1]
        ax5.plot(angles, vals_r, color=STYLE['accent1'], lw=2)
        ax5.fill(angles, vals_r, color=STYLE['accent1'], alpha=0.2)
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(cats, color=STYLE['text'], size=9)
        ax5.set_ylim(0, 1)
        ax5.set_facecolor(STYLE['panel'])
        ax5.spines['polar'].set_color(STYLE['grid'])
        ax5.yaxis.set_tick_params(labelcolor=STYLE['grid'])
        ax5.set_title("Radar des Métriques", color=STYLE['accent1'],
                      fontweight='bold', pad=15)

        # ── 6. Temps par agent ────────────────────────────────────
        ax6 = fig.add_subplot(gs[1, 2])
        agent_names = list(agent_timings.keys())
        agent_times = list(agent_timings.values())
        colors6 = [STYLE['accent1'], STYLE['accent2'], STYLE['accent3'],
                   STYLE['accent4'], '#C77DFF'][:len(agent_names)]
        bars = ax6.barh(agent_names, agent_times, color=colors6, edgecolor='none')
        for bar, val in zip(bars, agent_times):
            ax6.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f'{val:.1f}s', va='center', ha='left', fontsize=9)
        ax6.set_title("Temps d'Exécution par Agent", color=STYLE['accent4'], fontweight='bold')
        ax6.set_xlabel("Secondes"); ax6.grid(True, axis='x')

        # ── 7. Messages inter-agents ──────────────────────────────
        ax7 = fig.add_subplot(gs[2, 0])
        channels = list(bus_stats['channels'].keys())
        counts   = list(bus_stats['channels'].values())
        colors7  = plt.cm.plasma(np.linspace(0.2, 0.9, len(channels)))
        wedges, texts, autotexts = ax7.pie(
            counts, labels=channels, autopct='%1.0f%%',
            colors=colors7, textprops={'fontsize': 7, 'color': STYLE['text']},
            startangle=140
        )
        for at in autotexts:
            at.set_color('white'); at.set_fontsize(8)
        ax7.set_title("Messages Inter-Agents", color=STYLE['accent3'], fontweight='bold')

        # ── 8. Statistiques structurelles des formules ────────────
        ax8 = fig.add_subplot(gs[2, 1])
        stats_list = preprocess_result.get('stats_list', [])
        if stats_list:
            ratios = [s['ratio'] for s in stats_list]
            ax8.hist(ratios, bins=30, color=STYLE['accent4'], edgecolor='none', alpha=0.85)
            ax8.axvline(np.mean(ratios), color=STYLE['accent2'], ls='--',
                        lw=2, label=f'μ={np.mean(ratios):.2f}')
            ax8.set_title("Ratio Clauses/Variables", color=STYLE['accent4'], fontweight='bold')
            ax8.set_xlabel("n_clauses / n_vars"); ax8.set_ylabel("Fréquence")
            ax8.legend(); ax8.grid(True)

        # ── 9. Temps de résolution symbolique ─────────────────────
        ax9 = fig.add_subplot(gs[2, 2])
        solve_times = symbolic_result.get('solve_times', [])
        sv_arr      = np.array(solve_times)
        if sv_arr.max() > 0:
            ax9.hist(sv_arr * 1000, bins=30, color=STYLE['accent1'],
                     edgecolor='none', alpha=0.85)
            ax9.set_title("Temps Résolution PySAT (ms)", color=STYLE['accent1'], fontweight='bold')
            ax9.set_xlabel("ms"); ax9.set_ylabel("Fréquence"); ax9.grid(True)
        else:
            ax9.text(0.5, 0.5, "Labels pré-calculés\n(instances synthétiques)",
                     ha='center', va='center', transform=ax9.transAxes,
                     color=STYLE['text'], fontsize=11)
            ax9.set_title("Temps Résolution PySAT", color=STYLE['accent1'], fontweight='bold')

        # ── 10. Architecture MAS (diagramme textuel) ──────────────
        ax10 = fig.add_subplot(gs[3, :])
        ax10.set_xlim(0, 10); ax10.set_ylim(0, 3)
        ax10.axis('off')
        ax10.set_title("Architecture du Système Multi-Agent",
                       color=STYLE['accent1'], fontweight='bold', pad=10)

        agents_info = [
            (1.0,  1.5, "A0\nOrchestrator", STYLE['accent4']),
            (3.2,  2.2, "A1\nPréprocesseur", STYLE['accent1']),
            (3.2,  0.8, "A2\nSolveur\nSymbolique", STYLE['accent2']),
            (6.0,  1.5, "A3\nTransformer\n(GPU)", STYLE['accent3']),
            (8.8,  1.5, "A4\nÉvaluateur", '#C77DFF'),
        ]
        box_w, box_h = 1.4, 0.9

        for (x, y, label, col) in agents_info:
            rect = mpatches.FancyBboxPatch(
                (x - box_w/2, y - box_h/2), box_w, box_h,
                boxstyle="round,pad=0.08",
                facecolor=col + '33', edgecolor=col, linewidth=2.5,
                transform=ax10.transData
            )
            ax10.add_patch(rect)
            ax10.text(x, y, label, ha='center', va='center',
                      color='white', fontsize=8, fontweight='bold',
                      linespacing=1.4)

        # Flèches
        arrows = [
            (1.7, 1.5, 2.5, 2.2),   # A0 → A1
            (1.7, 1.5, 2.5, 0.8),   # A0 → A2
            (3.95, 2.2, 5.3, 1.7),  # A1 → A3
            (3.95, 0.8, 5.3, 1.3),  # A2 → A3
            (6.7, 1.5, 8.1, 1.5),   # A3 → A4
        ]
        for (x1, y1, x2, y2) in arrows:
            ax10.annotate("", xy=(x2, y2), xytext=(x1, y1),
                          arrowprops=dict(arrowstyle='->', color=STYLE['grid'],
                                         lw=2, connectionstyle='arc3,rad=0.1'))

        # Bus de messages
        ax10.text(5.0, 0.2, "── MessageBus (communication synchrone) ──",
                  ha='center', va='center', fontsize=9,
                  color=STYLE['grid'], style='italic')

        # Tableau récapitulatif métriques
        met_text = (
            f"  Accuracy={acc:.3f}   Precision={prec:.3f}   "
            f"Recall={rec:.3f}   F1={f1:.3f}   AUC={auc:.3f}  "
            f"  |  Messages total={bus_stats['total_messages']}"
        )
        ax10.text(5.0, 2.85, met_text, ha='center', va='center',
                  fontsize=9, color=STYLE['accent4'],
                  bbox=dict(facecolor=STYLE['panel'], edgecolor=STYLE['accent4'],
                            boxstyle='round,pad=0.4', lw=1.5))

        plt.savefig('mas_sat_report.png', dpi=130, bbox_inches='tight',
                    facecolor=STYLE['bg'])
        plt.show()
        print("\n  ✓ Figure sauvegardée : mas_sat_report.png")


# ═════════════════════════════════════════════════════════════
#  SECTION 9 — AGENT A0 : ORCHESTRATEUR
# ═════════════════════════════════════════════════════════════

class OrchestratorAgent(BaseAgent):
    """
    Agent A0 — Orchestrateur
    ─────────────────────────
    Point d'entrée du pipeline MAS.
    Il instancie, coordonne et séquence tous les autres agents.

    Pipeline :
      1. Chargement des données
      2. A1 PreprocessorAgent   → tokenisation + stats
      3. A2 SymbolicSolverAgent → labels ground truth
      4. Split train/val/test
      5. A3 TransformerAgent    → entraînement & prédictions
      6. A4 EvaluatorAgent      → métriques & graphiques
      7. Rapport final
    """

    def run(
        self,
        use_synthetic: bool = True,
        satlib_dir:    str  = '/kaggle/working/generated_3sat',
        n_vars:        int  = 20,
        n_clauses:     int  = 85,
        n_instances:   int  = 2000,
    ):
        t_global = time.time()
        print("\n" + "╔" + "═"*68 + "╗")
        print("║" + "  MAS-SAT — SYSTÈME MULTI-AGENT POUR LA RÉSOLUTION SAT".center(68) + "║")
        print("╚" + "═"*68 + "╝\n")

        # ── Étape 0 : Données ──────────────────────────────────────
        self.log("Étape 0 — Chargement / Génération des données")
        if use_synthetic:
            instances = generate_balanced_dataset(
                n_vars=n_vars, n_clauses=n_clauses,
                n_sat=n_instances // 2, n_unsat=n_instances // 2,
                seed=SEED
            )
        else:
            instances = load_satlib(satlib_dir, max_instances=n_instances)
            # Vérifier équilibre
            counts = Counter()
            for f in instances:
                with SATSolver(name='g3') as s:
                    s.append_formula(f.clauses)
                    counts[s.solve()] += 1
            if counts[True] == 0 or counts[False] == 0:
                self.log("Dataset déséquilibré — fallback synthétique")
                instances = generate_balanced_dataset(
                    n_vars=n_vars, n_clauses=n_clauses,
                    n_sat=n_instances // 2, n_unsat=n_instances // 2,
                    seed=SEED
                )

        # ── Étape 1 : A1 Préprocesseur ─────────────────────────────
        self.log("Étape 1 — Agent A1 : Préprocesseur")
        a1 = PreprocessorAgent(self.bus)
        prep_result = a1.run(instances)
        self.send("A1-Preprocessor", "task", "ack")

        token2id  = prep_result['token2id']
        tokenized = prep_result['tokenized']   # List[Tensor]

        # ── Étape 2 : A2 Solveur Symbolique ───────────────────────
        self.log("Étape 2 — Agent A2 : Solveur Symbolique")
        a2 = SymbolicSolverAgent(self.bus)
        sym_result = a2.run(instances)
        labels     = sym_result['labels']

        # ── Étape 3 : Split dataset ────────────────────────────────
        self.log("Étape 3 — Split train/val/test (80/10/10)")
        items = list(zip(tokenized, labels))   # [(Tensor, int)]
        idx   = torch.randperm(len(items), generator=torch.Generator().manual_seed(SEED))
        n     = len(items)
        n_tr  = int(0.80 * n)
        n_vl  = int(0.10 * n)

        train_data = [items[i] for i in idx[:n_tr]]
        val_data   = [items[i] for i in idx[n_tr:n_tr + n_vl]]
        test_data  = [items[i] for i in idx[n_tr + n_vl:]]
        self.log(f"  train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

        # ── Étape 4 : A3 Transformer ───────────────────────────────
        self.log("Étape 4 — Agent A3 : Transformer Neuronal")
        a3 = TransformerAgent(self.bus, vocab_size=prep_result['vocab_size'],
                              pad_id=token2id['PAD'])
        trans_result = a3.run(train_data, val_data, test_data)

        # ── Étape 5 : A4 Évaluateur ───────────────────────────────
        self.log("Étape 5 — Agent A4 : Évaluateur")

        agent_timings = {
            'A1-Preprocessor':  a1.elapsed,
            'A2-SymbolicSolver': a2.elapsed,
            'A3-Transformer':   a3.elapsed,
        }

        a4 = EvaluatorAgent(self.bus)
        final_metrics = a4.run(
            transformer_result=trans_result,
            symbolic_result=sym_result,
            preprocess_result=prep_result,
            agent_timings=agent_timings,
            bus_stats=self.bus.stats(),
        )
        agent_timings['A4-Evaluator'] = a4.elapsed

        # ── Rapport global ─────────────────────────────────────────
        t_total = time.time() - t_global
        print("\n" + "╔" + "═"*68 + "╗")
        print("║" + "  RAPPORT FINAL MAS-SAT".center(68) + "║")
        print("╠" + "═"*68 + "╣")
        print(f"║  Instances totales   : {n:<46}║")
        print(f"║  Vocabulaire         : {prep_result['vocab_size']:<46}║")
        print(f"║  Distribution labels : {str(sym_result['distribution']):<46}║")
        print(f"║  GPU(s) utilisés     : {torch.cuda.device_count():<46}║")
        print("╠" + "═"*68 + "╣")
        print(f"║  Accuracy            : {final_metrics['accuracy']:.4f}{' '*41}║")
        print(f"║  F1-score            : {final_metrics['f1']:.4f}{' '*41}║")
        print(f"║  ROC-AUC             : {final_metrics['auc']:.4f}{' '*41}║")
        print("╠" + "═"*68 + "╣")
        print(f"║  Temps total pipeline: {t_total:.2f}s{' '*41}║")
        bus = self.bus.stats()
        print(f"║  Messages échangés   : {bus['total_messages']:<46}║")
        print("╚" + "═"*68 + "╝")

        # Sauvegarde modèle
        model_core = a3.model.module if isinstance(a3.model, nn.DataParallel) else a3.model
        torch.save({
            'model_state_dict': model_core.state_dict(),
            'token2id':         token2id,
            'vocab_size':       prep_result['vocab_size'],
            'metrics':          final_metrics,
            'config': {
                'hidden_dim': 128, 'n_heads': 4,
                'n_layers': 3,     'ff_dim':  512,
                'max_len':  MAX_LEN, 'dropout': 0.1,
            }
        }, 'mas_sat_model.pt')
        print("\n  ✓ Modèle sauvegardé : mas_sat_model.pt")

        return final_metrics


# ═════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Configuration du run ───────────────────────────────────
    USE_SYNTHETIC = True          # False = charger depuis SATLIB
    SATLIB_DIR    = '/kaggle/working/generated_3sat'
    N_INSTANCES   = 2000          # nombre total d'instances (50/50 SAT/UNSAT)
    N_VARS        = 20            # variables par formule (synthétique)
    N_CLAUSES     = 85            # clauses par formule  (ratio ≈4.25 → zone de phase)

    bus           = MessageBus()
    orchestrator  = OrchestratorAgent("A0-Orchestrator", bus)

    orchestrator.run(
        use_synthetic = USE_SYNTHETIC,
        satlib_dir    = SATLIB_DIR,
        n_vars        = N_VARS,
        n_clauses     = N_CLAUSES,
        n_instances   = N_INSTANCES,
    )