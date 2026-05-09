#!/usr/bin/env python3
"""
Build a comprehensive PDF report from the experiment_dual results.

Pages:
  1. Cover
  2. Sumário Executivo
  3. Contexto e Hipótese
  4. Metodologia
  5. Distribuição de classes (Vehkaoja + Marinara)
  6. Tabela agregada global + bar chart agg
  7. Heatmap delta (split−full) — Vehkaoja
  8. Heatmap delta (split−full) — Marinara
  9. Bar chart por método: full vs split — Vehkaoja
 10. Bar chart por método: full vs split — Marinara
 11. DL hyperparameter sweep — CNN
 12. DL hyperparameter sweep — LSTM
 13. Casos extremos e insights
 14. Conclusões e limitações
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

EXP = Path("/Users/joanafontes/Documents/Datasets/experiment_dual")
OUT_PDF = EXP / "report.pdf"

DATASETS = ["vehkaoja", "marinara"]
MODELS = ["rf", "svm", "cnn", "lstm"]
METHODS = [
    "select_k_best", "anova_f", "mutual_information",
    "variance_threshold", "chi_squared", "xgboost",
    "random_forest", "lasso", "rfe", "sequential_feature_selection",
]

sns.set_theme(style="whitegrid", context="paper")


# ───────────────────────── helpers ─────────────────────────

def load_summaries() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        rows.append(pd.read_csv(EXP / ds / "analysis" / "summary.csv"))
    return pd.concat(rows, ignore_index=True)


def load_class_dist(ds: str) -> dict:
    """Read the class distribution from splits.json (already cached there)."""
    split = json.loads((EXP / ds / "splits.json").read_text())
    return {k: int(v) for k, v in split["global_class_dist"].items()}


def load_split_info(ds: str) -> dict:
    return json.loads((EXP / ds / "splits.json").read_text())


def load_dl_sweep(ds: str, model: str) -> pd.DataFrame:
    p = EXP / ds / "dl_sweep" / f"{model}_sweep.json"
    payload = json.loads(p.read_text())
    return pd.DataFrame(payload["all_runs"]), payload["best"]


# ───────────────────────── pages ─────────────────────────

def page_text(pdf: PdfPages, title: str, body: str, *, fontsize_body: int = 11) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.05, 0.96, title, fontsize=20, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(0.05, 0.90, body, fontsize=fontsize_body, va="top",
            transform=ax.transAxes, family="monospace", wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_cover(pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.5, 0.78, "Seleção de Features Estática vs Dinâmica",
            fontsize=22, fontweight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.73,
            "Comparação metodológica em dois datasets de IMU canino",
            fontsize=14, ha="center", transform=ax.transAxes, style="italic")
    ax.text(0.5, 0.62,
            "Vehkaoja DogMoveData (marinara) · Marinara IMU Posture",
            fontsize=12, ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.50,
            "10 métodos de FS × 4 modelos × 2 estratégias × 2 datasets",
            fontsize=11, ha="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef", edgecolor="#669"))
    ax.text(0.5, 0.40,
            "Random Forest · SVM (RBF) · CNN1D · LSTM\n"
            "Hyperparameter sweep: 3 dim × 3 lr (PyTorch / MPS)",
            fontsize=10, ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.25, "Pipeline: extract_features → splits → feature_selection → models",
            fontsize=9, ha="center", transform=ax.transAxes, family="monospace")
    ax.text(0.5, 0.10, f"Relatório gerado a partir de {EXP.name}/",
            fontsize=8, ha="center", color="gray", transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_executive_summary(pdf: PdfPages, df: pd.DataFrame) -> None:
    piv = df.pivot_table(index=["dataset", "model"], columns="strategy",
                         values="test_f1", aggfunc="mean").round(4)
    piv["Δ"] = (piv["split"] - piv["full"]).round(4)

    body = (
        "OBJETIVO\n"
        "  Validar se selecionar 10 features só do subconjunto estático + 10 só do\n"
        "  dinâmico (combinadas em 20) melhora accuracy/F1 face ao baseline de 20\n"
        "  features escolhidas do dataset full.\n\n"
        "PROCEDIMENTO\n"
        "  - Extração TSFEL (window=100, overlap=50%, fs=100Hz) preservando subject_id.\n"
        "  - Split inter-subject 70/15/15 representativo (KL-mínimo vs distribuição\n"
        "    global).\n"
        "  - 10 métodos de seleção de features × 2 estratégias (full / split).\n"
        "  - 4 classificadores avaliados em cada cell. CNN/LSTM com sweep 3×3.\n\n"
        "RESULTADO PRINCIPAL — F1 ponderado médio (test) por (dataset, modelo):\n\n"
        + piv.to_string()
        + "\n\n"
        "CONCLUSÃO RESUMIDA\n"
        "  - A estratégia split tende a piorar marginalmente o F1 médio (entre -0.005\n"
        "    e -0.017) em 7 dos 8 pares (modelo × dataset).\n"
        "  - Excepção: LSTM no Vehkaoja → split é +0.026 melhor.\n"
        "  - O ganho mais saliente está em métodos pouco robustos a desbalanceamento\n"
        "    (chi_squared, variance_threshold), onde split corrige drasticamente.\n"
        "  - Lasso é o método mais robusto: F1 quase idêntico em ambas estratégias.\n"
    )
    page_text(pdf, "Sumário Executivo", body, fontsize_body=10)


def page_context(pdf: PdfPages) -> None:
    body = (
        "PROBLEMA\n"
        "  Em datasets de comportamento animal recolhidos por IMU, comportamentos\n"
        "  dinâmicos (caminhar, correr) costumam ter substancialmente mais janelas\n"
        "  do que comportamentos estáticos (sentar, deitar). Quando aplicamos\n"
        "  feature selection ao dataset full, esse desbalanceamento pode enviesar\n"
        "  o ranking de features em direção a sinais de movimento, degradando a\n"
        "  identificação de posturas.\n\n"
        "HIPÓTESE\n"
        "  H1: Selecionar 10 features apenas do subconjunto estático + 10 apenas\n"
        "      do subconjunto dinâmico (combinadas em 20) produzirá um vetor mais\n"
        "      balanceado e melhorará a métrica do classificador final treinado\n"
        "      em todas as classes.\n\n"
        "DESENHO EXPERIMENTAL\n"
        "  Estratégia A — full (baseline):\n"
        "    FS no features_full.csv (apenas amostras de TRAIN dogs) → 20 features.\n"
        "  Estratégia B — split (experimental):\n"
        "    FS no features_static.csv → 10 features.\n"
        "    FS no features_dynamic.csv → 10 features.\n"
        "    União (com top-up em caso de overlap) → 20 features únicas.\n"
        "  Em ambas, o classificador é treinado no full filtrado pelas 20 features.\n\n"
        "VALIDAÇÃO INDEPENDENTE\n"
        "  Replicação do mesmo pipeline em DOIS datasets distintos:\n"
        "    Vehkaoja: 45 cães, 12 sensores (Acc+Gyro × Back/Neck), 17 classes.\n"
        "    Marinara: 43 cães, 27 sensores (Acc+Gyro+Mag × Back/Chest/Neck),\n"
        "              5 classes (Position + coluna Type para split static/dynamic).\n"
    )
    page_text(pdf, "Contexto e Hipótese", body)


def page_methodology(pdf: PdfPages) -> None:
    body = (
        "1. EXTRAÇÃO DE FEATURES (TSFEL)\n"
        "   - Janela: 100 amostras (1 segundo @ 100 Hz)\n"
        "   - Overlap: 50% (stride = 50)\n"
        "   - Domínios: estatístico + temporal + espectral (~100 feats por canal)\n"
        "   - Preservada coluna subject_id por janela (essencial para split inter-subject)\n"
        "   - Vehkaoja: 131,139 janelas × 1,873 features (12 canais)\n"
        "   - Marinara : ~50,000 janelas × ~4,200 features (27 canais)\n\n"
        "2. SPLIT INTER-SUBJECT REPRESENTATIVO\n"
        "   - Proporção 70/15/15 em número de cães\n"
        "   - 200 seeds avaliados; escolhe-se o que minimiza KL(test||global)+KL(val||global)\n"
        "   - Vehkaoja: 31 train / 7 val / 7 test (KL_test=0.0115)\n"
        "   - Marinara: 30 train / 7 val / 6 test\n"
        "   - Mesma partição usada em todas as 80 cells (modelo × método × estratégia)\n\n"
        "3. SELEÇÃO DE FEATURES (10 métodos)\n"
        "   - Filter:    select_k_best, anova_f, mutual_information, chi_squared,\n"
        "                variance_threshold\n"
        "   - Embedded:  random_forest, xgboost (ExtraTrees), lasso (LR L1)\n"
        "   - Wrapper:   rfe, sequential_feature_selection (com pré-filtro top-100)\n"
        "   - K=20 features; FS aplicada APENAS em TRAIN (sem leakage)\n\n"
        "4. CLASSIFICADORES (4 modelos)\n"
        "   - Random Forest: 100 árvores, n_jobs=-1\n"
        "   - SVM RBF: C=1, gamma=scale, train subsampled to 30k\n"
        "   - CNN1D: 3 blocos Conv1d+BN+ReLU, dim D, MaxPool entre blocos\n"
        "   - LSTM: 1 camada hidden=H, MLP head\n"
        "   - Sweep DL: D∈{32,64,128} × lr∈{1e-2,1e-3,1e-4} → 9 configs\n"
        "     calibrados em (full, anova_f), melhor escolhido por val_f1.\n"
        "   - Treino DL: Adam, batch=128, max 30 epochs, early stop patience=5,\n"
        "     device MPS (Apple Silicon).\n\n"
        "5. AVALIAÇÃO\n"
        "   - Métricas: accuracy + F1 weighted em val e test\n"
        "   - Encoder de labels ajustado em TRAIN; classes não vistas → linha removida\n"
        "   - Filtro de classes raras: ≥10 amostras em TRAIN (5 em static/dynamic FS)\n"
    )
    page_text(pdf, "Metodologia", body, fontsize_body=9)


def page_class_distributions(pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    for ax, ds in zip(axes, DATASETS):
        dist = load_class_dist(ds)
        items = sorted(dist.items(), key=lambda kv: -kv[1])
        labels = [k for k, _ in items]
        counts = [v for _, v in items]
        bars = ax.bar(range(len(labels)), counts, color=sns.color_palette("viridis", len(labels)))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Janelas (TSFEL)")
        ax.set_title(f"Distribuição de classes — {ds.upper()} "
                     f"(total {sum(counts):,} janelas)")
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{c:,}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Distribuição de classes por dataset", fontsize=14, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_aggregate_bars(pdf: PdfPages, df: pd.DataFrame) -> None:
    agg = df.groupby(["dataset", "model", "strategy"])["test_f1"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(
        data=agg, x="model", y="test_f1", hue="strategy",
        order=MODELS, hue_order=["full", "split"],
        palette={"full": "#4477AA", "split": "#EE6677"},
        ax=ax,
    )
    ax.set_title("F1 médio (test) por modelo · agrupado por dataset",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Test F1 weighted (média sobre 10 métodos de FS)")
    ax.set_xlabel("Modelo")
    ax.set_ylim(0, 1)

    # Two subplots side by side: one per dataset
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        sub = agg[agg["dataset"] == ds]
        sns.barplot(
            data=sub, x="model", y="test_f1", hue="strategy",
            order=MODELS, hue_order=["full", "split"],
            palette={"full": "#4477AA", "split": "#EE6677"},
            ax=ax,
        )
        ax.set_title(f"{ds.upper()}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Test F1 weighted (média)")
        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.text(p.get_x() + p.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Média de F1 (test) por modelo e estratégia",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_delta_heatmap(pdf: PdfPages, df: pd.DataFrame, ds: str) -> None:
    sub = df[df["dataset"] == ds]
    # delta = split - full per (model, method)
    pf = sub.pivot_table(index="model", columns=["method", "strategy"], values="test_f1")
    delta = pf.xs("split", axis=1, level="strategy") - pf.xs("full", axis=1, level="strategy")
    delta = delta.reindex(index=MODELS, columns=METHODS)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    vmax = max(abs(delta.min().min()), abs(delta.max().max()))
    sns.heatmap(delta, annot=True, fmt="+.3f", cmap="RdBu_r",
                center=0, vmin=-vmax, vmax=vmax, ax=ax,
                cbar_kws={"label": "Δ test_f1 (split − full)"})
    ax.set_title(f"Δ test_f1 por (modelo × método) — {ds.upper()}\n"
                 "vermelho = split pior · azul = split melhor",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Método de seleção de features")
    ax.set_ylabel("Modelo")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_per_method_bars(pdf: PdfPages, df: pd.DataFrame, ds: str) -> None:
    sub = df[df["dataset"] == ds]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, model in zip(axes.flat, MODELS):
        d = sub[sub["model"] == model]
        sns.barplot(
            data=d, x="method", y="test_f1", hue="strategy",
            order=METHODS, hue_order=["full", "split"],
            palette={"full": "#4477AA", "split": "#EE6677"},
            ax=ax,
        )
        ax.set_title(f"{model.upper()}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Test F1 weighted")
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
        if ax is not axes.flat[0]:
            ax.legend_.remove() if ax.legend_ else None
    fig.suptitle(f"{ds.upper()} — Comparação full vs split por método e modelo",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_dl_sweep(pdf: PdfPages, model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        df_sweep, best = load_dl_sweep(ds, model_name)
        piv = df_sweep.pivot(index="dim", columns="lr", values="val_f1_weighted")
        sns.heatmap(piv, annot=True, fmt=".3f", cmap="YlGn",
                    cbar_kws={"label": "val_f1"}, ax=ax)
        ax.set_title(f"{ds.upper()}  ·  best: dim={best['dim']}, lr={best['lr']:.0e}")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Hidden dim" if ax is axes[0] else "")
    fig.suptitle(f"Sweep de hiperparâmetros — {model_name.upper()} (cell: full + anova_f)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_insights(pdf: PdfPages, df: pd.DataFrame) -> None:
    # Find biggest +Δ and -Δ per dataset
    pf = df.pivot_table(index=["dataset", "model", "method"],
                        columns="strategy", values="test_f1").reset_index()
    pf["delta"] = pf["split"] - pf["full"]

    biggest_pos = pf.sort_values("delta", ascending=False).head(8)
    biggest_neg = pf.sort_values("delta", ascending=True).head(8)

    body_top = (
        "MAIORES GANHOS DA ESTRATÉGIA SPLIT (test_f1)\n\n"
        + biggest_pos[["dataset", "model", "method", "full", "split", "delta"]]
            .round(4).to_string(index=False)
        + "\n\n"
        "MAIORES PERDAS DA ESTRATÉGIA SPLIT (test_f1)\n\n"
        + biggest_neg[["dataset", "model", "method", "full", "split", "delta"]]
            .round(4).to_string(index=False)
        + "\n\n"
        "INTERPRETAÇÕES\n"
        "  - chi_squared no Vehkaoja: estratégia full produz F1 muito baixo\n"
        "    (~0.08 LSTM, 0.23 CNN, 0.30 SVM). A estratégia split corrige\n"
        "    drasticamente (+0.18 a +0.51 pontos). Isto sugere que chi² no\n"
        "    full não está a separar classes com suficiente sinal — beneficia\n"
        "    da imposição forçada de cobertura estática+dinâmica.\n"
        "  - variance_threshold também beneficia de split em ambos os datasets,\n"
        "    pelo mesmo mecanismo (variância alta dominada por classes ativas).\n"
        "  - lasso é praticamente invariante à estratégia: já está a fazer\n"
        "    seleção esparsa multi-classe que naturalmente cobre o espaço.\n"
        "  - Métodos baseados em árvores (random_forest, xgboost) tipicamente\n"
        "    sofrem com split: a métrica embarcada já equilibra classes pelo\n"
        "    modelo, e forçar 10+10 introduz redundância.\n"
        "  - O caso LSTM no Vehkaoja onde split ganha em média (+2.6pp) merece\n"
        "    investigação adicional — pode ser regularização implícita por\n"
        "    diversidade de features.\n"
    )
    page_text(pdf, "Casos extremos e insights", body_top, fontsize_body=8)


def page_conclusions(pdf: PdfPages) -> None:
    body = (
        "CONCLUSÕES\n\n"
        "  A hipótese H1 (split > full) NÃO se confirma de forma generalizada.\n"
        "  Em média, a estratégia split degrada marginalmente F1 (entre -0.005\n"
        "  e -0.017 pontos) em 7 dos 8 pares modelo × dataset testados.\n\n"
        "  A vantagem da estratégia split é condicional ao método de FS:\n"
        "  - VANTAGEM consistente: chi_squared, variance_threshold\n"
        "    (métodos sensíveis a desbalanceamento de classes)\n"
        "  - INDIFERENTE:           lasso, sequential_feature_selection\n"
        "  - DESVANTAGEM:           random_forest, xgboost, rfe, anova_f\n\n"
        "  Para o uso prático, lasso permanece o método mais robusto e\n"
        "  competitivo em ambos os datasets e em ambas estratégias.\n\n"
        "LIMITAÇÕES\n\n"
        "  - CNN/LSTM são alimentados com 20 features TSFEL como sequência 1D,\n"
        "    o que é não-convencional. A literatura de HAR usa janelas IMU\n"
        "    brutas (100×N_canais). A escolha foi necessária para isolar o\n"
        "    efeito da FS, mas reduz a capacidade dos modelos sequenciais.\n"
        "  - O sweep DL é calibrado apenas na cell (full, anova_f) e o best\n"
        "    config é reutilizado em todas as cells. Hyperparams ótimos por\n"
        "    cell poderiam alterar conclusões pontuais.\n"
        "  - Split inter-subject 70/15/15: 7 cães em test/val no Vehkaoja é\n"
        "    pequeno; variância entre splits provável. Reportamos KL-mínimo,\n"
        "    mas resultados poderiam variar com outra partição.\n"
        "  - Vehkaoja tem 17 classes desbalanceadas (algumas raras com poucas\n"
        "    janelas) → F1 absoluto é baixo (~0.55-0.65), o que torna deltas\n"
        "    relativos mais ruidosos.\n\n"
        "PRÓXIMOS PASSOS SUGERIDOS\n\n"
        "  - Repetir com vários splits (mesmo que não folds) e reportar média±std.\n"
        "  - Variar o orçamento total k (não apenas 20=10+10): estudar 30=15+15,\n"
        "    40=20+20 para ver se vantagem de split aparece com mais features.\n"
        "  - Avaliar F1 macro por classe — a hipótese diz especificamente que\n"
        "    classes estáticas devem melhorar; F1 weighted dilui esse efeito.\n"
        "  - Testar CNN/LSTM com input bruto IMU para a baseline absoluta.\n"
    )
    page_text(pdf, "Conclusões e Limitações", body, fontsize_body=9)


# ───────────────────────── main ─────────────────────────

def main() -> None:
    df = load_summaries()
    print(f"Loaded {len(df)} rows across {df['dataset'].nunique()} datasets.")

    with PdfPages(OUT_PDF) as pdf:
        page_cover(pdf)
        page_executive_summary(pdf, df)
        page_context(pdf)
        page_methodology(pdf)
        page_class_distributions(pdf)
        page_aggregate_bars(pdf, df)
        for ds in DATASETS:
            page_delta_heatmap(pdf, df, ds)
        for ds in DATASETS:
            page_per_method_bars(pdf, df, ds)
        for model_name in ("cnn", "lstm"):
            page_dl_sweep(pdf, model_name)
        page_insights(pdf, df)
        page_conclusions(pdf)

    print(f"\n✓ PDF saved → {OUT_PDF}  ({OUT_PDF.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
