import pandas as pd


def csv_to_latex_tables(csv_path):
    df = pd.read_csv(csv_path)

    df = df[df["Image"] != "Average"]

    df["Image_Num"] = df["Image"].str.extract("(\d+)").astype(int)

    acc_table = df.pivot(
        index="Image_Num", columns="Prompt Folder", values="Pixel Accuracy"
    )
    iou_table = df.pivot(index="Image_Num", columns="Prompt Folder", values="IoU")

    acc_table = acc_table.map(lambda x: f"{x:.3f}")
    iou_table = iou_table.map(lambda x: f"{x:.3f}")

    latex_acc = acc_table.to_latex(
        caption="Pixel Accuracy by Image and Prompt",
        label="tab:pixel_accuracy",
        position="h!",
        column_format="l" + "c" * len(acc_table.columns),
    )

    latex_iou = iou_table.to_latex(
        caption="IoU by Image and Prompt",
        label="tab:iou",
        position="h!",
        column_format="l" + "c" * len(iou_table.columns),
    )

    avg_acc = df["Pixel Accuracy"].mean()
    avg_iou = df["IoU"].mean()

    latex_acc = latex_acc.replace(
        r"\bottomrule",
        f"\\midrule\nAverage & {' & '.join([f'{avg_acc:.3f}']*len(acc_table.columns))} \\\\\n\\bottomrule",
    )

    latex_iou = latex_iou.replace(
        r"\bottomrule",
        f"\\midrule\nAverage & {' & '.join([f'{avg_iou:.3f}']*len(iou_table.columns))} \\\\\n\\bottomrule",
    )

    return latex_acc, latex_iou


csv_path = "experiments/results/segmentation_accuracy_results.csv"
latex_acc, latex_iou = csv_to_latex_tables(csv_path)

print("Pixel Accuracy Table:")
print(latex_acc)
print("\nIoU Table:")
print(latex_iou)
