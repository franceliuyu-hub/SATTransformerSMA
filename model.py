

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import to_dense_batch
import math


class GraphEncoder(nn.Module):
    """
    Encodeur GNN léger pour embeddings initiaux des nœuds du graphe bipartite.
    Variables et clauses partagent le même espace latent.
    """
    def __init__(self, node_feature_dim=3, hidden_dim=256, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Projection initiale
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        
        # Couches de message passing
        self.conv_layers = nn.ModuleList([
            SAGEConvBlock(hidden_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x, edge_index, edge_attr=None):
        x = self.node_encoder(x)
        
        for conv, ln in zip(self.conv_layers, self.layer_norms):
            x_new = conv(x, edge_index, edge_attr)
            x = ln(x + x_new)  # Residual + LayerNorm
        
        return x


class SAGEConvBlock(MessagePassing):
    """
    Bloc GraphSAGE pour message passing sur graphe bipartite.
    Agrégation mean-neighborhood.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='mean')
        self.lin_l = nn.Linear(in_dim, out_dim)  # Pour le nœud central
        self.lin_r = nn.Linear(in_dim, out_dim)  # Pour les voisins
        self.activation = nn.ReLU()
        
    def forward(self, x, edge_index, edge_attr=None):
        # x est un tenseur unique, pas un tuple
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr=None):
        # x_j: features des voisins
        return self.lin_r(x_j)
    
    def update(self, aggr_out, x):
        # x: features du nœud central (pas un tuple)
        # Combine nœud courant + agrégation voisins
        return self.activation(self.lin_l(x) + aggr_out)


class MultiAgentAttention(nn.Module):
    """
    Couche d'attention multi-têtes spécialisées par type d'agent.
    
    Têtes 0-1: Variable -> Clause (positif)
    Têtes 2-3: Variable -> Clause (négatif)  
    Têtes 4-5: Clause -> Variable (demande)
    Têtes 6-7: Variable -> Variable (coopération)
    """
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        assert num_heads == 8, "Architecture conçue pour 8 têtes spécialisées"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # 32
        
        # Projections Q, K, V par tête
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Masques de spécialisation (apprentissage des rôles)
        self.head_specialization = nn.Parameter(
            torch.randn(num_heads, 4)  # 4 types d'interaction
        )
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
        # Pour stocker les poids (visualisation)
        self.attention_weights = None
        
    def forward(self, x, node_types, edge_index, batch):
        # ... Q, K, V ...
        
        # Attention scores: (batch_size, num_heads, num_nodes, num_nodes)
        # OU (num_nodes, num_nodes, num_heads) selon einsum
        
        # CORRECT: utiliser matmul pour batch + heads
        Q = Q.transpose(0, 1)  # (H, N, D)
        K = K.transpose(0, 1)  # (H, N, D)
        V = V.transpose(0, 1)  # (H, N, D)
        
        # Scores: (H, N, N)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Masque: (N, N) -> (1, N, N) -> broadcast sur H
        base_mask = self._build_graph_mask(node_types, edge_index, x.size(0))
        attn_mask = base_mask.unsqueeze(0)  # (1, N, N)
        
        # Application masque
        attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
        
        # Softmax sur dimension N
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Output: (H, N, D)
        out = torch.matmul(attn_weights, V)
        
        # Retour: (N, H, D) -> (N, H*D)
        out = out.transpose(0, 1).reshape(-1, self.hidden_dim)
        
        return out
    
    def _build_graph_mask(self, node_types, edge_index, num_nodes):
        """
        Construit le masque d'attention restreint au graphe bipartite.
        Retourne un masque de shape (num_nodes, num_nodes).
        """
        device = node_types.device
        
        # Masque de base: (num_nodes, num_nodes) booléen
        mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)
        
        # Connexions du graphe
        src, dst = edge_index
        mask[src, dst] = True
        
        return mask


class AgentPolicy(nn.Module):
    """
    Politique de décision pour chaque agent-variable.
    Output: distribution sur {TRUE, FALSE, UNDECIDED}.
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # TRUE, FALSE, UNDECIDED
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Valeur de l'état
        )
        
    def forward(self, x_var):
        """
        Args:
            x_var: (N_vars, hidden_dim) embeddings des agents-variables
        Returns:
            logits: (N_vars, 3) politique
            value: (N_vars, 1) estimation de valeur
        """
        logits = self.policy_net(x_var)
        value = self.value_net(x_var)
        return logits, value


class SMATransformer(nn.Module):
    """
    Architecture complète: SMA-Transformer pour résolution SAT.
    
    Pipeline:
    1. Encodage graphe (GNN)
    2. Communication multi-agent (Transformer spécialisé)
    3. Décision par agent (Policy)
    4. Hybridation optionnelle avec MiniSat
    """
    def __init__(
        self,
        node_feature_dim=3,
        hidden_dim=256,
        num_gnn_layers=3,
        num_transformer_layers=4,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. Encodeur graphe
        self.graph_encoder = GraphEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers
        )
        
        # 2. Couches Transformer multi-agent
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiAgentAttention(hidden_dim, num_heads, dropout),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                ),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim)
            })
            for _ in range(num_transformer_layers)
        ])
        
        # 3. Politique des agents
        self.agent_policy = AgentPolicy(hidden_dim)
        
        # 4. Classification globale (optionnel, pour supervision)
        self.global_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Probabilité SAT
        )
        
        # Pour stocker les états intermédiaires (explicabilité)
        self.communication_logs = []
        
    def forward(self, data, return_attention=False):
        """
        Args:
            data: Batch PyTorch Geometric Data
                - x: node features (N, node_feature_dim)
                - edge_index: arêtes du graphe (2, E)
                - edge_attr: optional edge features
                - node_types: 0=Var, 1=Clause (N,)
                - batch: index de batch (N,)
                - num_vars: nombre de variables par instance
        Returns:
            decisions: (N_vars, 3) logits des agents-variables
            sat_prob: (batch_size, 1) probabilité globale SAT
            attention_maps: liste des poids d'attention par couche
        """
        x, edge_index = data.x, data.edge_index
        node_types, batch = data.node_types, data.batch
        
        # 1. Encodage graphe
        x = self.graph_encoder(x, edge_index, getattr(data, 'edge_attr', None))
        
        # 2. Communication multi-agent (Transformer)
        attention_maps = []
        self.communication_logs = []
        
        for i, layer in enumerate(self.transformer_layers):
            # Attention
            x_attn = layer['attention'](x, node_types, edge_index, batch)
            x = layer['norm1'](x + x_attn)
            
            # Stockage pour explicabilité
            if return_attention:
                attention_maps.append(layer['attention'].attention_weights)
                self.communication_logs.append({
                    'layer': i,
                    'head_weights': layer['attention'].attention_weights
                })
            
            # Feed-forward
            x_ffn = layer['ffn'](x)
            x = layer['norm2'](x + x_ffn)
        
        # 3. Séparation variables / clauses
        var_mask = (node_types == 0)
        clause_mask = (node_types == 1)
        
        x_vars = x[var_mask]      # (N_vars, hidden_dim)
        x_clauses = x[clause_mask]  # (N_clauses, hidden_dim)
        
        # 4. Décision des agents-variables
        policy_logits, state_values = self.agent_policy(x_vars)
        
        # 5. Prédiction globale (pour supervision)
        # Pooling sur les variables de chaque instance
        sat_prob = self.global_classifier(x_vars)
        sat_prob = torch.sigmoid(sat_prob)
        
        return {
            'policy_logits': policy_logits,      # Décisions agents
            'state_values': state_values,         # Valeurs d'état (RL)
            'sat_prob': sat_prob,                 # Probabilité globale
            'var_embeddings': x_vars,             # Pour visualisation
            'clause_embeddings': x_clauses,       # Pour visualisation
            'attention_maps': attention_maps if return_attention else None
        }
    
    def get_warm_start(self, data, threshold=0.6):
        """
        Génère une affectation initiale pour MiniSat.
        Prend les décisions les plus confiantes des agents.
        
        Returns:
            assignment: dict {var_id: value} ou None si pas assez confiant
        """
        with torch.no_grad():
            output = self.forward(data)
        
        logits = output['policy_logits']  # (N_vars, 3)
        probs = F.softmax(logits, dim=-1)  # (N_vars, 3)
        
        # Sélection des décisions confiantes
        max_probs, decisions = probs.max(dim=-1)
        
        assignment = {}
        for i, (prob, dec) in enumerate(zip(max_probs, decisions)):
            if prob > threshold and dec != 2:  # Pas UNDECIDED
                var_id = i + 1  # Variables indexées à 1
                value = (dec == 0)  # TRUE=0, FALSE=1 dans notre encodage
                assignment[var_id] = value
        
        return assignment if len(assignment) > data.num_vars * 0.5 else None


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def build_cnf_graph(cnf_formula, max_var=None):
    """
    Convertit une formule CNF (PySAT) en graphe PyTorch Geometric.
    
    Args:
        cnf_formula: Instance pysat.formula.CNF
        max_var: Nombre maximum de variables (pour padding)
    
    Returns:
        data: torch_geometric.data.Data
    """
    from torch_geometric.data import Data
    
    clauses = cnf_formula.clauses
    n_vars = max_var or max(abs(lit) for clause in clauses for lit in clause)
    n_clauses = len(clauses)
    
    # Nœuds: [variables, clauses]
    # Features: [type, degree, normalized_index]
    node_features = []
    node_types = []
    
    # Variables (indices 0 à n_vars-1)
    for i in range(n_vars):
        node_features.append([
            0.0,  # type: variable
            0.0,  # degree (calculé après)
            i / n_vars  # index normalisé
        ])
        node_types.append(0)
    
    # Clauses (indices n_vars à n_vars+n_clauses-1)
    for i, clause in enumerate(clauses):
        node_features.append([
            1.0,  # type: clause
            len(clause),  # degree initial
            i / n_clauses  # index normalisé
        ])
        node_types.append(1)
    
    # Arêtes: littéraux connectent variables à clauses
    edge_index = []
    edge_attr = []  # 1.0 pour positif, -1.0 pour négatif
    
    for c_idx, clause in enumerate(clauses):
        clause_node = n_vars + c_idx
        for lit in clause:
            var_node = abs(lit) - 1  # 0-indexed
            edge_index.append([var_node, clause_node])
            edge_attr.append(1.0 if lit > 0 else -1.0)
            # Arête bidirectionnelle pour message passing
            edge_index.append([clause_node, var_node])
            edge_attr.append(1.0 if lit > 0 else -1.0)
    
    # Calcul des degrés réels des variables
    var_degrees = [0] * n_vars
    for e in edge_index:
        if e[0] < n_vars:
            var_degrees[e[0]] += 1
    
    for i in range(n_vars):
        node_features[i][1] = var_degrees[i] / max(var_degrees)
    
    # Conversion en tenseurs
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    node_types = torch.tensor(node_types, dtype=torch.long)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_types=node_types,
        num_vars=n_vars,
        num_clauses=n_clauses
    )
    
    return data

def generate_random_cnf(n_vars, n_clauses, seed=None):
    """Générateur CNF aléatoire compatible python-sat"""
    if seed is not None:
        random.seed(seed)
    
    cnf = CNF()
    for _ in range(n_clauses):
        # 3-SAT aléatoire
        clause = random.sample(range(1, n_vars + 1), 3)
        clause = [v if random.choice([True, False]) else -v for v in clause]
        cnf.append(clause)
    
    return cnf

if __name__ == '__main__':
    from pysat.formula import CNF
    from pysat.formula import CNF
    from pysat.solvers import Solver
    import random
    
    # Test sur petite instance
    print("=" * 60)
    print("TEST SMA-TRANSFORMER")
    print("=" * 60)
    
    # Génération instance test
    cnf = generate_random_cnf(n_vars=10, n_clauses=43, seed=42)
    
    # Construction graphe
    data = build_cnf_graph(cnf)
    print(f"\nGraphe construit:")
    print(f"  Variables: {data.num_vars}")
    print(f"  Clauses: {data.num_clauses}")
    print(f"  Nœuds: {data.x.size(0)}")
    print(f"  Arêtes: {data.edge_index.size(1)}")
    
    # Batch singleton
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    
    # Modèle
    model = SMATransformer(
        node_feature_dim=3,
        hidden_dim=256,
        num_gnn_layers=3,
        num_transformer_layers=4,
        num_heads=8
    )
    
    # Forward
    output = model(data, return_attention=True)
    
    print(f"\nOutput:")
    print(f"  Policy logits: {output['policy_logits'].shape}")
    print(f"  SAT prob: {output['sat_prob'].item():.4f}")
    print(f"  Attention maps: {len(output['attention_maps'])} couches")
    
    # Warm-start
    assignment = model.get_warm_start(data, threshold=0.6)
    print(f"\nWarm-start assignment: {len(assignment) if assignment else 0} variables assignées")
    
    print("\n✓ Test réussi")