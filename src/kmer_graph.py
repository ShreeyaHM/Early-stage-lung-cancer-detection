"""
Genomic K-Mer Graph Representation & Topological Feature Extraction
Constructs directed k-mer transition network graphs from DNA sequences,
computing topological graph metrics and network visualizations.
"""

import itertools
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple
from collections import Counter

def extract_kmers(sequence: str, k: int = 3) -> List[str]:
    """Extract overlapping k-mers of length k from a cleaned DNA sequence."""
    seq = sequence.upper().strip()
    if len(seq) < k:
        return []
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def build_kmer_graph(sequence: str, k: int = 3) -> nx.DiGraph:
    """
    Build a directed transition graph where nodes are k-mers,
    and edges represent sequential transitions (k-mer_i -> k-mer_{i+1}) with transition counts as weights.
    """
    kmers = extract_kmers(sequence, k=k)
    G = nx.DiGraph()
    
    if not kmers:
        return G
        
    for kmer in kmers:
        if not G.has_node(kmer):
            G.add_node(kmer, count=0)
        G.nodes[kmer]['count'] += 1
        
    for i in range(len(kmers) - 1):
        src, dst = kmers[i], kmers[i+1]
        if G.has_edge(src, dst):
            G[src][dst]['weight'] += 1
        else:
            G.add_edge(src, dst, weight=1)
            
    return G

def compute_kmer_graph_metrics(G: nx.DiGraph) -> Dict[str, float]:
    """
    Computes topological properties of the k-mer transition graph.
    Somatic mutation hotspots introduce altered pathway traversals, increasing graph entropy
    and altering degree distribution.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    if num_nodes <= 1:
        return {
            'num_nodes': float(num_nodes),
            'num_edges': float(num_edges),
            'graph_density': 0.0,
            'avg_degree': 0.0,
            'transition_entropy': 0.0,
            'clustering_coef': 0.0,
            'spectral_radius': 0.0
        }
        
    density = nx.density(G)
    degrees = [d for _, d in G.degree()]
    avg_deg = float(np.mean(degrees)) if degrees else 0.0
    
    # Transition entropy across outgoing edges
    entropies = []
    for node in G.nodes():
        out_edges = G.out_edges(node, data='weight')
        weights = [w for _, _, w in out_edges]
        tot_w = sum(weights)
        if tot_w > 0:
            probs = np.array(weights) / tot_w
            ent = -np.sum(probs * np.log2(probs + 1e-12))
            entropies.append(ent)
            
    transition_entropy = float(np.mean(entropies)) if entropies else 0.0
    
    # Average clustering coefficient (undirected projection)
    try:
        clustering_coef = float(nx.average_clustering(G.to_undirected()))
    except Exception:
        clustering_coef = 0.0
        
    # Spectral radius (largest eigenvalue of adjacency matrix)
    try:
        adj = nx.to_numpy_array(G)
        eigenvals = np.linalg.eigvals(adj)
        spectral_radius = float(np.max(np.abs(eigenvals)))
    except Exception:
        spectral_radius = 0.0
        
    return {
        'num_nodes': float(num_nodes),
        'num_edges': float(num_edges),
        'graph_density': round(density, 4),
        'avg_degree': round(avg_deg, 2),
        'transition_entropy': round(transition_entropy, 4),
        'clustering_coef': round(clustering_coef, 4),
        'spectral_radius': round(spectral_radius, 4)
    }

def create_kmer_graph_plot(G: nx.DiGraph, max_nodes: int = 40) -> go.Figure:
    """
    Generate an interactive Plotly network diagram of the top k-mer transition graph.
    """
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="No K-Mer Graph Available")
        return fig
        
    # Subgraph of top nodes by count if too large
    if G.number_of_nodes() > max_nodes:
        sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[1].get('count', 0), reverse=True)
        top_node_names = [n for n, _ in sorted_nodes[:max_nodes]]
        sub_G = G.subgraph(top_node_names).copy()
    else:
        sub_G = G
        
    pos = nx.spring_layout(sub_G, seed=42, k=0.3)
    
    # Edge trace
    edge_x = []
    edge_y = []
    for edge in sub_G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.0, color='rgba(148, 163, 184, 0.6)'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Node trace
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_colors = []
    
    for node in sub_G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        count = sub_G.nodes[node].get('count', 1)
        deg = sub_G.degree(node)
        node_text.append(f"<b>K-Mer:</b> {node}<br>Occurrences: {count}<br>Connections: {deg}")
        node_sizes.append(min(12 + count * 2, 28))
        node_colors.append(deg)
        
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[n for n in sub_G.nodes()],
        textposition="top center",
        textfont=dict(size=9, color="#1e293b"),
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='Blues',
            reversescale=False,
            color=node_colors,
            size=node_sizes,
            colorbar=dict(
                thickness=12,
                title=dict(text="Node Degree", side="top"),
                xanchor="left"
            ),
            line_width=1.5,
            line_color='#2563eb'
        )
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text="<b>Genomic K-Mer Sequence Graph Topology</b>", font=dict(size=14)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=10, r=10, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(248,250,252,0.6)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=380
        )
    )
    return fig
