"""PNG: visualize the credit-risk Bayesian Network DAG."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import IMPROVED_EDGES


def main(out: Path) -> int:
    G = nx.DiGraph()
    G.add_edges_from(IMPROVED_EDGES)

    # Manually-tuned hierarchical layout for clarity
    layers = {
        "Evaluacion crediticia historica": (0, 4),
        "Edad": (-2, 4),
        "Evaluacion Sueldo": (-4, 4),
        "Genero": (-2, 3),
        "Residencia": (-1, 2),
        "Proposito": (-4, 2),
        "Nivel de ahorro": (3, 4),
        "Monto del credito": (-2, 1),
        "Duracion": (1, 1),
        "Riesgo": (0, -1),
    }
    pos = {n: layers.get(n, (0, 0)) for n in G.nodes()}

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=110)

    # Color the target node differently
    node_colors = ["#dc2626" if n == "Riesgo" else "#6366f1" for n in G.nodes()]
    sizes = [3500 if n == "Riesgo" else 2200 for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, edge_color="#94a3b8", arrows=True,
                           arrowsize=18, width=1.4, node_size=2200,
                           connectionstyle="arc3,rad=0.05", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=sizes,
                           edgecolors="white", linewidths=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8.5, font_color="white", font_weight="bold", ax=ax)

    ax.set_title("Bayesian Network — credit risk DAG (Hill-Climb + expert refinement)",
                 fontsize=13, weight="bold")
    ax.text(0.5, -0.03,
            "Each arrow encodes a conditional probability. P(Riesgo | evidence) is "
            "computed via belief propagation when evidence is observed.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color="#475569")
    ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).parents[2] / "portfolio-website" / "public" / "previews" / "credit-risk.png"
    )
    sys.exit(main(out))
