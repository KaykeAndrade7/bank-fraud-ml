"""
Reporting module — gera um PDF profissional com:
 - Cabeçalho (título, data, autor)
 - Sumário das métricas (tabela estilizada)
 - Box com modelo escolhido
 - Gráficos (importados de reports/figures)
 - Conclusão automática
 - Rodapé com paginação

Uso:
    from src.reporting import generate_pdf_report
    generate_pdf_report(df_tuned, model_name="XGBoost (Tuned)")
"""

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
)
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from datetime import datetime
import os
import glob
import pandas as pd
from typing import Optional

# ---------- Configurações ----------
DEFAULT_OUTPUT = "reports/model_report.pdf"
FIGURES_DIR = "reports/plots"
PAGE_WIDTH, PAGE_HEIGHT = A4
AUTHOR_NAME = "Kayke Andrade"  # troque se quiser

# ---------- Helpers de estilo ----------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleLarge", fontSize=20, leading=24, spaceAfter=12, alignment=1))  # centered
styles.add(ParagraphStyle(name="Heading", fontSize=14, leading=18, spaceAfter=8, textColor=colors.HexColor("#003366")))
styles.add(ParagraphStyle(name="NormalSmall", fontSize=10, leading=12))
styles.add(ParagraphStyle(name="TableHeader", fontSize=10, leading=12, alignment=1, textColor=colors.white))
styles.add(ParagraphStyle(name="Footer", fontSize=8, leading=10, alignment=1, textColor=colors.grey))

# ---------- Footer / Page numbering ----------
def _footer(canvas_obj: canvas.Canvas, doc):
    canvas_obj.saveState()
    footer_text = f"Relatório gerado por {AUTHOR_NAME} — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    page_num_text = f"Página {doc.page}"
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.setFillColor(colors.grey)
    canvas_obj.drawCentredString(PAGE_WIDTH/2.0, 12 * mm / 2.0, footer_text)
    canvas_obj.drawRightString(PAGE_WIDTH - 12 * mm, 12 * mm / 2.0, page_num_text)
    canvas_obj.restoreState()

# ---------- Construção de tabela estilizada ----------
def build_metrics_table(df: pd.DataFrame):
    """
    Espera um DataFrame com colunas: model, roc_auc, recall, precision
    Retorna um reportlab Table com estilo aplicado.
    """
    # Cabeçalho
    cols = ["Modelo", "ROC-AUC", "Recall", "Precision"]
    data = [cols]

    # Linhas formatadas
    for _, row in df.iterrows():
        data.append([
            str(row.get("model", row.get("Model", ""))),
            f"{row.get('roc_auc', 0):.4f}",
            f"{row.get('recall', 0):.4f}",
            f"{row.get('precision', 0):.4f}"
        ])

    # Largura das colunas (em mm convertidos para points)
    col_widths = [100 * mm, 30 * mm, 30 * mm, 30 * mm]

    table = Table(data, colWidths=col_widths, hAlign="LEFT")

    # Estilo
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),  # cabeçalho escuro
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
    ])

    # linhas zebra
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.add("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F5F5F5"))

    table.setStyle(style)
    return table

# ---------- Conclusão automática ----------
def build_conclusion(df: pd.DataFrame):
    """
    Retorna um parágrafo de conclusão analisando qual modelo tem melhor roc_auc, recall e precision.
    df deve ter colunas: model, roc_auc, recall, precision
    """
    best_auc = df.loc[df["roc_auc"].idxmax()]["model"]
    best_recall = df.loc[df["recall"].idxmax()]["model"]
    best_precision = df.loc[df["precision"].idxmax()]["model"]

    text = (
        f"O modelo com melhor ROC-AUC foi <b>{best_auc}</b>.\n\n"
        f"O modelo com maior Recall foi <b>{best_recall}</b>.\n\n"
        f"O modelo mais preciso foi <b>{best_precision}</b>.\n\n"
        "Observação: a escolha do modelo final deve considerar o trade-off entre Recall e Precision "
        "conforme o caso de uso de negócios. Em cenários de fraude, costuma-se priorizar Recall."
    )
    return Paragraph(text, styles["NormalSmall"])

# ---------- Buscar figuras disponíveis ----------
def _collect_figures(figures_dir=FIGURES_DIR):
    if not os.path.exists(figures_dir):
        return []
    paths = sorted(glob.glob(os.path.join(figures_dir, "*.png")))
    return paths

# ---------- Principal: geração do PDF ----------
def generate_pdf_report(metrics_df: pd.DataFrame,
                        model_name: Optional[str] = None,
                        figures_dir: str = FIGURES_DIR,
                        output_path: str = DEFAULT_OUTPUT):
    """
    Gera o PDF completo.
    - metrics_df: DataFrame com colunas (model, roc_auc, recall, precision)
    - model_name: string opcional para destacar modelo final
    - figures_dir: pasta com .png (roc_auc_comparison.png, recall_comparison.png, etc)
    - output_path: caminho do PDF final
    """

    # Garantir pastas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Normalizar DataFrame (garantir colunas esperadas)
    df = metrics_df.copy()
    # deprecated column names handling
    if "model" not in df.columns and "Model" in df.columns:
        df = df.rename(columns={"Model": "model"})
    # garantir colunas numéricas
    for c in ["roc_auc", "recall", "precision"]:
        if c not in df.columns:
            df[c] = 0.0
    # garantir ordem desejada
    df = df[["model", "roc_auc", "recall", "precision"]]

    # Documento
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=18 * mm, leftMargin=18 * mm,
                            topMargin=18 * mm, bottomMargin=24 * mm)

    story = []

    # --- Capa ---
    story.append(Paragraph("Credit Fraud Detection — Model Report", styles["TitleLarge"]))
    subtitle = "Análise comparativa de modelos de Machine Learning"
    story.append(Paragraph(subtitle, styles["Heading"]))
    story.append(Spacer(1, 6))

    gen_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    meta = f"Gerado em: {gen_date} — Autor: {AUTHOR_NAME}"
    story.append(Paragraph(meta, styles["NormalSmall"]))
    story.append(Spacer(1, 12))

    # mini resumo do dataset
    story.append(Paragraph("<b>Resumo do dataset</b>", styles["Heading"]))
    story.append(Paragraph(
        "Dataset: Kaggle — Credit Card Fraud Detection. 284.807 transações. "
        "Classes altamente desbalanceadas (≈0.17% fraudes). Features V1–V28 (PCA) + Amount + Time.",
        styles["NormalSmall"]
    ))
    story.append(Spacer(1, 8))

    # --- Seção: Pré-processamento ---
    story.append(Paragraph("Pré-processamento", styles["Heading"]))
    story.append(Paragraph(
        "Foram aplicados split 80/20 estratificado, normalização (StandardScaler) no conjunto de treino, "
        "e balanceamento com SMOTE aplicado somente ao treino. Os artefatos (scaler + arrays) foram salvos em data/processed e models/.",
        styles["NormalSmall"]
    ))
    story.append(Spacer(1, 10))

    # --- Seção: Métricas e Tabela ---
    story.append(Paragraph("Comparação de métricas", styles["Heading"]))
    story.append(Spacer(1, 6))
    table = build_metrics_table(df)
    story.append(KeepTogether(table))
    story.append(Spacer(1, 12))

    # --- Box: Modelo escolhido (se fornecido) ou sugerido por AUC ---
    if model_name:
        story.append(Paragraph("Modelo selecionado para produção", styles["Heading"]))
        story.append(Paragraph(f"<b>{model_name}</b>", styles["NormalSmall"]))
        story.append(Spacer(1, 8))
    else:
        # sugerir por AUC
        best_auc_model = df.loc[df["roc_auc"].idxmax()]["model"]
        story.append(Paragraph("Sugestão de modelo para produção (baseado em ROC-AUC)", styles["Heading"]))
        story.append(Paragraph(f"<b>{best_auc_model}</b>", styles["NormalSmall"]))
        story.append(Spacer(1, 8))

    # --- Seção: Gráficos ---
    story.append(PageBreak())
    story.append(Paragraph("Gráficos comparativos", styles["Heading"]))
    story.append(Spacer(1, 8))

    figures = _collect_figures(figures_dir)
    if not figures:
        story.append(Paragraph("Nenhum gráfico encontrado em reports/figures. Salve PNGs com nomes como 'roc_auc_comparison.png'.", styles["NormalSmall"]))
    else:
        for fig in figures:
            # limitar largura e manter aspecto
            try:
                img = Image(fig, width=160 * mm, height=None)
                story.append(img)
                caption = os.path.basename(fig).replace("_", " ").replace(".png", "")
                story.append(Paragraph(f"Figura: {caption}", styles["NormalSmall"]))
                story.append(Spacer(1, 10))
            except Exception as e:
                story.append(Paragraph(f"Erro ao incluir figura {fig}: {e}", styles["NormalSmall"]))
                story.append(Spacer(1, 6))

    # --- Seção: Conclusão automática ---
    story.append(PageBreak())
    story.append(Paragraph("Conclusão automática", styles["Heading"]))
    story.append(Spacer(1, 6))
    story.append(build_conclusion(df))
    story.append(Spacer(1, 12))

    # --- Rodapé final e build ---
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)

    print(f"✔ Relatório PDF salvo em: {output_path}")
    return output_path
