from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import os
import pandas as pd


def generate_pdf_report(metrics_df):
    os.makedirs("reports", exist_ok=True)

    # Criar o documento PDF
    doc = SimpleDocTemplate(
        "reports/model_report.pdf",
        pagesize=A4,
        title="Credit Fraud Detection — Model Report"
    )

    styles = getSampleStyleSheet()
    story = []

    # ----------------------------------
    # 📌 CAPA
    # ----------------------------------
    title = Paragraph("<b>Credit Fraud Detection — Model Report</b>", styles["Title"])
    subtitle = Paragraph("Análise comparativa de modelos de Machine Learning", styles["Heading2"])
    date = Paragraph("Relatório gerado automaticamente", styles["BodyText"])

    story.append(title)
    story.append(Spacer(1, 12))
    story.append(subtitle)
    story.append(Spacer(1, 20))
    story.append(date)
    story.append(PageBreak())

    # ----------------------------------
    # 🟩 SEÇÃO 1 — INTRODUÇÃO
    # ----------------------------------
    intro_title = Paragraph("<b>1 — Introdução</b>", styles["Heading1"])
    intro_text = Paragraph(
        """
        Este relatório apresenta a comparação entre diferentes modelos de Machine Learning
        treinados para o problema de detecção de fraudes em transações bancárias. 
        O dataset utilizado é altamente desbalanceado e contém variáveis obtidas por PCA. 
        Antes do treinamento, foi realizado pré-processamento incluindo normalização e SMOTE.
        """,
        styles["BodyText"]
    )

    story.append(intro_title)
    story.append(Spacer(1, 8))
    story.append(intro_text)
    story.append(PageBreak())

    # ----------------------------------
    # 🟧 SEÇÃO 2 — TABELA DE MÉTRICAS
    # ----------------------------------
    table_title = Paragraph("<b>2 — Tabela de Métricas</b>", styles["Heading1"])
    story.append(table_title)
    story.append(Spacer(1, 12))

    data = [["Modelo", "ROC-AUC", "Recall", "Precision"]]

    # ✔ Correção: iterar pelas LINHAS
    for model_name, row in metrics_df.iterrows():
        data.append([
            model_name,
            f"{row['roc_auc']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['precision']:.4f}"
        ])

    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 1, colors.black)
    ]))

    story.append(table)
    story.append(PageBreak())

    # ----------------------------------
    # 🟨 SEÇÃO 3 — GRÁFICOS
    # ----------------------------------
    graph_title = Paragraph("<b>3 — Gráficos Comparativos</b>", styles["Heading1"])
    story.append(graph_title)
    story.append(Spacer(1, 12))

    for img in [
        "reports/roc_auc_comparison.png",
        "reports/recall_comparison.png",
        "reports/precision_comparison.png"
    ]:
        if os.path.exists(img):
            story.append(Image(img, width=400, height=250))
            story.append(Spacer(1, 18))
        else:
            story.append(Paragraph(f"[ERRO] Arquivo não encontrado: {img}", styles["BodyText"]))

    story.append(PageBreak())

    # ----------------------------------
    # 🟪 SEÇÃO 4 — CONCLUSÃO AUTOMÁTICA
    # ----------------------------------
    conclusion_title = Paragraph("<b>4 — Conclusão Automática</b>", styles["Heading1"])
    story.append(conclusion_title)
    story.append(Spacer(1, 12))

    # ✔ Encontra automaticamente os melhores
    best_auc = metrics_df["roc_auc"].idxmax()
    best_recall = metrics_df["recall"].idxmax()
    best_precision = metrics_df["precision"].idxmax()

    conclusion_text = Paragraph(
        f"""
        Com base nas métricas avaliadas, podemos concluir:
        <br/><br/>
        • O modelo com melhor <b>ROC-AUC</b> foi <b>{best_auc}</b>. <br/>
        • O modelo com melhor <b>Recall</b> foi <b>{best_recall}</b>. <br/>
        • O modelo com melhor <b>Precision</b> foi <b>{best_precision}</b>. <br/><br/>
        Cada modelo apresenta características diferentes, o que abre espaço para abordagens 
        avançadas como ensembles ou tuning de hiperparâmetros.
        """,
        styles["BodyText"]
    )

    story.append(conclusion_text)

    # ----------------------------------
    # ✔ Construir o PDF
    # ----------------------------------
    doc.build(story)
    print("✔ Relatório PDF salvo em: reports/model_report.pdf")
