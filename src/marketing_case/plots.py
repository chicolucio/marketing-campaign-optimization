import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy.stats import ks_2samp

CURRENT_COLOR_CYCLER = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def customized_pairplot(
    dataframe: pd.DataFrame,
    columns: list[str],
    hue_column: str = None,
    alpha: float = 0.5,
    corner: bool = True,
    palette: str = None,
    common_norm: bool = False,
) -> None:
    """
    Customized pairplot for a subset of columns in a dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe.
    columns : list[str]
        Columns to include in the pairplot.
    hue_column : str, optional
        Column to use for coloring, by default None.
    alpha : float, optional
        Transparency of the points, by default 0.5.
    corner : bool, optional
        If True, only plot the lower triangle, by default True.
    palette : str, optional
        Palette to use for the hue column, by default None.
    common_norm : bool, optional
        If True, the KDE plots will be normalized, by default False.
    """

    if not palette:
        palette = (
            CURRENT_COLOR_CYCLER[: len(dataframe[hue_column].unique())]
            if hue_column
            else CURRENT_COLOR_CYCLER[:1]
        )

    analysis = columns.copy() + [hue_column] if hue_column else columns

    sns.pairplot(
        dataframe[analysis],
        hue=hue_column,
        diag_kind="kde",
        plot_kws={"alpha": alpha},
        diag_kws={"common_norm": common_norm},
        corner=corner,
        palette=palette if hue_column else None,
    )

    plt.show()


def plot_category_distribution_by_cluster(
    dataframe: pd.DataFrame,
    columns: list[str],
    rows_cols: tuple = (2, 3),
    figsize: tuple = (15, 8),
    column_cluster: str = "cluster",
) -> None:
    """
    Plot how different categories of a column are distributed within each cluster.
    Clusters are on the x-axis and the variable of interest is used as hue.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe.
    columns : list[str]
        Columns to include in the plots.
    rows_cols : tuple, optional
        Number of rows and columns in the grid of axes, by default (2, 3).
    figsize : tuple, optional
        Width and height of the figure, by default (15, 8).
    column_cluster : str, optional
        Name of the column with the cluster labels, by default "cluster".
    """

    fig, axs = plt.subplots(
        nrows=rows_cols[0], ncols=rows_cols[1], figsize=figsize, sharey=True
    )

    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)

    for ax, col in zip(axs.flatten(), columns):
        h = sns.histplot(
            x=column_cluster,
            hue=col,
            data=dataframe,
            ax=ax,
            multiple="fill",
            stat="percent",
            discrete=True,
            shrink=0.8,
        )

        n_clusters = dataframe[column_cluster].nunique()
        h.set_xticks(range(n_clusters))
        h.yaxis.set_major_formatter(PercentFormatter(1))
        h.set_ylabel("")
        h.tick_params(axis="both", which="both", length=0)

        for bars in h.containers:
            h.bar_label(
                bars,
                label_type="center",
                labels=[f"{b.get_height():.1%}" for b in bars],
                color="white",
                weight="bold",
                fontsize=11,
            )

        for bar in h.patches:
            bar.set_linewidth(0)

        sns.move_legend(
            h, "lower center", bbox_to_anchor=(0.5, 1.0), ncol=dataframe[col].nunique()
        )

    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    plt.show()


def plot_cluster_distribution_by_category(
    dataframe: pd.DataFrame,
    columns: list[str],
    rows_cols: tuple = (2, 3),
    figsize: tuple = (15, 8),
    column_cluster: str = "cluster",
) -> None:
    """
    Plot how clusters distribute within the unique values of each column.
    The variable of interest is on the x-axis and the clusters are used as hue.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe.
    columns : list[str]
        Columns to include in the plots.
    rows_cols : tuple, optional
        Number of rows and columns in the grid of axes, by default (2, 3).
    figsize : tuple, optional
        Width and height of the figure, by default (15, 8).
    column_cluster : str, optional
        Name of the column with the cluster labels, by default "cluster".
    """
    fig, axs = plt.subplots(
        nrows=rows_cols[0], ncols=rows_cols[1], figsize=figsize, sharey=True
    )

    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)

    for ax, col in zip(axs.flatten(), columns):
        h = sns.histplot(
            x=col,
            hue=column_cluster,
            data=dataframe,
            ax=ax,
            multiple="fill",
            stat="percent",
            discrete=True,
            shrink=0.8,
        )

        if dataframe[col].dtype != "object":
            h.set_xticks(range(dataframe[col].nunique()))

        h.yaxis.set_major_formatter(PercentFormatter(1))
        h.set_ylabel("")
        h.tick_params(axis="both", which="both", length=0)

        for bars in h.containers:
            h.bar_label(
                bars,
                label_type="center",
                labels=[f"{b.get_height():.1%}" for b in bars],
                color="white",
                weight="bold",
                fontsize=11,
            )

        for bar in h.patches:
            bar.set_linewidth(0)

        legend = h.get_legend()
        legend.remove()

    labels = [text.get_text() for text in legend.get_texts()]

    fig.legend(
        handles=legend.legend_handles,
        labels=labels,
        loc="upper center",
        ncols=dataframe[column_cluster].nunique(),
        title="Clusters",
        bbox_to_anchor=(0.5, 1.05),
    )

    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    plt.show()


def plot_coefficients(df_coefs: pd.DataFrame, title: str = "Coefficients") -> None:
    """
    Plot coefficients.

    Parameters
    ----------
    df_coefs : pd.DataFrame
        DataFrame with coefficients values. The labels should be in the index.
    title : str, optional
        Title of the plot, by default "Coefficients".
    """
    df_coefs.plot.barh(figsize=(10, 15))
    plt.title(title)
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coefficients")
    plt.gca().get_legend().remove()
    plt.show()


def plot_model_performance_metrics(model_performance_data: pd.DataFrame) -> None:
    """
    Plot model performance metrics.

    Parameters
    ----------
    model_performance_data : pd.DataFrame
        DataFrame with model performance metrics.
    """
    fig, axs = plt.subplots(4, 2, figsize=(9, 9), sharex=True)

    metrics_to_compare = [
        "time_seconds",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_roc_auc",
        "test_average_precision",
    ]

    metric_labels = [
        "Time (s)",
        "Accuracy",
        "Balanced Accuracy",
        "F1",
        "Precision",
        "Recall",
        "AUROC",
        "AUPRC",
    ]

    for ax, metric, label in zip(axs.flatten(), metrics_to_compare, metric_labels):
        sns.boxplot(
            x="model",
            y=metric,
            data=model_performance_data,
            ax=ax,
            showmeans=True,
        )
        ax.set_title(label, fontsize="small")
        ax.set_ylabel(label, fontsize="x-small")
        ax.tick_params(axis="x", rotation=90, labelsize="x-small")
        ax.tick_params(axis="y", labelsize="x-small")

    plt.tight_layout()

    plt.show()


def plot_ks_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
    """
    Generate a Kolmogorov-Smirnov (KS) plot to compare the distributions of predicted
    probabilities for the positive and negative classes in a binary classification
    problem.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    None
        Displays the KS plot.
    """

    # Separate probabilities into two groups: positives and negatives
    positives = y_pred_proba[y_true == 1]
    negatives = y_pred_proba[y_true == 0]

    # Compute empirical CDFs
    sorted_pos = np.sort(positives)
    sorted_neg = np.sort(negatives)

    cdf_pos = np.arange(1, len(sorted_pos) + 1) / len(sorted_pos)
    cdf_neg = np.arange(1, len(sorted_neg) + 1) / len(sorted_neg)

    # Compute the KS statistic
    ks_statistic, _ = ks_2samp(positives, negatives)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot CDFs
    ax.plot(sorted_pos, cdf_pos, label="Positive Class (y=1)", color="C0")
    ax.plot(sorted_neg, cdf_neg, label="Negative Class (y=0)", color="C1")

    # Find the maximum vertical distance (KS statistic)
    ks_x = np.linspace(0, 1, 1000)
    cdf_pos_interp = np.interp(ks_x, sorted_pos, cdf_pos, left=0, right=1)
    cdf_neg_interp = np.interp(ks_x, sorted_neg, cdf_neg, left=0, right=1)
    ks_diff = np.abs(cdf_pos_interp - cdf_neg_interp)
    max_ks_idx = np.argmax(ks_diff)
    max_ks_x = ks_x[max_ks_idx]
    max_ks_y1 = cdf_pos_interp[max_ks_idx]
    max_ks_y0 = cdf_neg_interp[max_ks_idx]

    # Highlight the KS statistic
    ax.vlines(
        max_ks_x,
        max_ks_y0,
        max_ks_y1,
        color="C2",
        linestyle="dashed",
        label=f"KS Statistic = {ks_statistic:.3f}",
    )
    ax.scatter([max_ks_x], [max_ks_y1], color="C0", zorder=3)
    ax.scatter([max_ks_x], [max_ks_y0], color="C1", zorder=3)

    # Labels, title, and legend
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Kolmogorov-Smirnov (KS) Plot")
    ax.legend()

    plt.show()
