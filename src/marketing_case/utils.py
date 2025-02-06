from functools import cached_property
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from scipy.stats import (
    chi2_contingency,
    kstest,
    levene,
    mannwhitneyu,
    norm,
    shapiro,
    ttest_ind,
    zscore,
)


def inspect_outliers(
    dataframe: pd.DataFrame, column: str, threshold: float = 1.5
) -> pd.DataFrame:
    """
    Inspect outliers in a dataframe column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe.
    column : str
        Column to inspect.
    threshold : float, optional
        Threshold for outlier detection, by default 1.5.

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers.
    """
    first_quartile = dataframe[column].quantile(0.25)
    third_quartile = dataframe[column].quantile(0.75)
    interquartile_range = third_quartile - first_quartile
    lower_bound = first_quartile - threshold * interquartile_range
    upper_bound = third_quartile + threshold * interquartile_range
    outliers = dataframe[
        (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
    ]
    return outliers


def best_grid_shape(n: int) -> tuple[int, int]:
    """
    Compute the best (rows, cols) arrangement for a given number of plots.

    The goal is to create a nearly square grid that efficiently fits all plots.

    Parameters
    ----------
    n : int
        Number of plots.

    Returns
    -------
    tuple[int, int]
        Optimal (rows, cols) grid size.
    """
    if n == 1:
        return (1, 1)

    rows = int(np.floor(np.sqrt(n)))
    cols = int(np.ceil(n / rows))

    return rows, cols


class StatisticalAnalysisBinary:
    """
    Perform statistical analysis on a binary target variable.

    This class provides methods for analyzing the distribution, normality,
    variance homogeneity, and differences between two groups defined by a
    binary target variable. It includes statistical tests such as:

    - Shapiro-Wilk test for normality
    - Levene's test for homogeneity of variances
    - Mann-Whitney U test for non-parametric comparisons
    - Independent t-test (Welch's by default)
    - Chi-squared test for independence

    Attributes
    ----------
    dataframe : pd.DataFrame
        The input dataset.
    target_column : str
        The binary target variable used for grouping.
    classes : tuple
        The class labels representing the two groups.
    """

    def __init__(
        self, dataframe: pd.DataFrame, target_column: str, classes: tuple = (0, 1)
    ) -> None:
        """
        Initialize the StatisticalAnalysisBinary object.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataset containing the target variable and independent features.
        target_column : str
            The name of the binary target variable.
        classes : tuple, optional
            The class labels representing the two groups, by default (0, 1).
        """
        self.dataframe = dataframe
        self.target_column = target_column
        self.classes = classes

    @cached_property
    def groupby(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Group the dataframe by the target column.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            DataFrames with the two groups.
        """

        grouping = self.dataframe.groupby(self.target_column, observed=False)
        first_group = grouping.get_group(self.classes[0])
        second_group = grouping.get_group(self.classes[1])

        return first_group, second_group

    def shape_analysis(self, columns: list[str]) -> pd.DataFrame:
        """
        Analyze the shape of the distribution of numerical columns.

        This method calculates skewness and kurtosis for the specified columns
        and classifies them based on their symmetry and tailedness.

        Skewness Classification:
        - "symmetric" if skewness is between -0.05 and 0.05.
        - "right-skewed" if skewness is greater than or equal to 0.05.
        - "left-skewed" if skewness is less than -0.05.

        Kurtosis Classification:
        - "mesokurtic" if kurtosis is between -0.05 and 0.05 (normal distribution).
        - "platykurtic" if kurtosis is less than -0.05 (flatter distribution).
        - "leptokurtic" if kurtosis is greater than 0.05 (more peaked distribution).

        Parameters
        ----------
        columns : list[str]
            List of numerical column names to analyze.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing:
            - "column": Name of the column.
            - "skewness": Skewness value.
            - "skewness_classification": Classification of skewness.
            - "kurtosis": Kurtosis value.
            - "kurtosis_classification": Classification of kurtosis.
        """
        # get skewness and kurtosis for each column
        results = []
        for column in columns:
            skewness = self.dataframe[column].skew()
            kurtosis = self.dataframe[column].kurtosis()

            # classify skewness as normal, right-skewed, or left-skewed
            skewness_classification = (
                "symmetric"
                if -0.05 < skewness < 0.05
                else ("right-skewed" if skewness >= 0.05 else "left-skewed")
            )
            # classify kurtosis as normal, platykurtic, or leptokurtic
            kurtosis_classification = (
                "mesokurtic"
                if -0.05 < kurtosis < 0.05
                else ("platykurtic" if kurtosis < -0.05 else "leptokurtic")
            )

            results.append(
                {
                    "column": column,
                    "skewness": skewness,
                    "skewness_classification": skewness_classification,
                    "kurtosis": kurtosis,
                    "kurtosis_classification": kurtosis_classification,
                }
            )
        return pd.DataFrame(results)

    def shapiro_wilk(self, columns: list[str], alpha: float = 0.05) -> pd.DataFrame:
        """
        Perform the Shapiro-Wilk test for normality.

        Parameters
        ----------
        columns : list[str]
            Columns to test.
        alpha : float, optional
            Significance level, by default 0.05.

        Returns
        -------
        pd.DataFrame
            DataFrame with the test results.
        """
        results = []
        for column in columns:
            statistic, p_value = shapiro(self.dataframe[column], nan_policy="omit")
            results.append(
                {
                    "column": column,
                    "statistic": statistic,
                    "p_value": p_value,
                    "normal": p_value > alpha,
                }
            )
        return pd.DataFrame(results)

    def kolmolgorov_smirnov(
        self,
        columns: list[str],
        alpha: float = 0.05,
        reference_cdf: Union[scipy.stats.rv_continuous, np.ndarray] = norm.cdf,
        mode: str = "auto",
    ) -> pd.DataFrame:
        """
        Perform the Kolmogorov-Smirnov test for normality.

        Parameters
        ----------
        columns : list[str]
            Columns to test.
        alpha : float, optional
            Significance level, by default 0.05.
        mode : str, optional
            Method to estimate the distribution, by default "auto".

        Returns
        -------
        pd.DataFrame
            DataFrame with the test results.
        """
        results = []
        for column in columns:
            zscore_column = zscore(self.dataframe[column], ddof=1, nan_policy="omit")
            statistic, p_value = kstest(zscore_column, reference_cdf, mode=mode)
            results.append(
                {
                    "column": column,
                    "statistic": statistic,
                    "p_value": p_value,
                    "normal": p_value > alpha,
                }
            )
        return pd.DataFrame(results)

    def levene(
        self, columns: list[str], alpha: float = 0.05, center="median"
    ) -> pd.DataFrame:
        """
        Perform the Levene test for homogeneity of variances.

        Parameters
        ----------
        columns : list[str]
            Columns to test.
        alpha : float, optional
            Significance level, by default 0.05.
        center : str, optional
            Center for the test, by default "median".

        Returns
        -------
        pd.DataFrame
            DataFrame with the test results.
        """
        results = []
        first_group, second_group = self.groupby
        for column in columns:
            statistic, p_value = levene(
                first_group[column],
                second_group[column],
                center=center,
            )
            results.append(
                {
                    "column": column,
                    "statistic": statistic,
                    "p_value": p_value,
                    "equal_variance": p_value > alpha,
                }
            )
        return pd.DataFrame(results)

    def mann_whitney(
        self, columns: list[str], alpha: float = 0.05, alternative: str = "two-sided"
    ) -> pd.DataFrame:
        """
        Perform the Mann-Whitney U test for independent samples.

        Parameters
        ----------
        columns : list[str]
            Columns to test.
        alpha : float, optional
            Significance level, by default 0.05.
        alternative : str, optional
            Alternative hypothesis, by default "two-sided".

        Returns
        -------
        pd.DataFrame
            DataFrame with the test results.
        """
        results = []
        first_group, second_group = self.groupby
        for column in columns:
            statistic, p_value = mannwhitneyu(
                first_group[column],
                second_group[column],
                alternative=alternative,
            )
            results.append(
                {
                    "column": column,
                    "statistic": statistic,
                    "p_value": p_value,
                    "different_distributions": p_value < alpha,
                }
            )
        return pd.DataFrame(results)

    def t_test(
        self,
        columns: list[str],
        alpha: float = 0.05,
        equal_var: bool = False,  # default to Welch's t-test. See below.
        alternative: str = "two-sided",
    ) -> pd.DataFrame:
        """
        Perform the t-test for independent samples.

        Parameters
        ----------
        columns : list[str]
            Columns to test.
        alpha : float, optional
            Significance level, by default 0.05.
        equal_var : bool, optional
            If True, perform the test assuming equal variances, by default False.
            The default is False, leading to the Welch's t-test.
            Welch's t-test does not assume equal population variance and it is
            more reliable when the two samples have unequal variances and/or
            unequal sample sizes. It is the default because it is as robust as
            the Student's t-test, which assumes equal variances, when the
            variances are equal. And it is not advisable anymore to perform the
            Levene test before the t-test, as shown by
            https://bpspsychub.onlinelibrary.wiley.com/doi/10.1348/000711004849222
        alternative : str, optional
            Alternative hypothesis, by default "two-sided".

        Returns
        -------
        pd.DataFrame
            DataFrame with the test results.
        """
        results = []
        first_group, second_group = self.groupby
        for column in columns:
            statistic, p_value = ttest_ind(
                first_group[column],
                second_group[column],
                equal_var=equal_var,
                alternative=alternative,
            )
            results.append(
                {
                    "column": column,
                    "statistic": statistic,
                    "p_value": p_value,
                    "different_means": p_value < alpha,
                }
            )
        return pd.DataFrame(results)

    def chi_squared(self, columns: list[str], alpha: float = 0.05) -> pd.DataFrame:
        """
        Perform the chi-squared test of independence.

        Parameters
        ----------
        columns : list[str]
            Columns to test.
        alpha : float, optional
            Significance level, by default 0.05.

        Returns
        -------
        pd.DataFrame
            DataFrame with the test results.
        """
        results = []
        for column in columns:
            less_than_5 = False
            contingency_table = pd.crosstab(
                self.dataframe[column], self.dataframe[self.target_column]
            )
            # check if any cell has an expected frequency < 5
            if contingency_table.values.min() < 5:
                less_than_5 = True

            statistic, p_value, _, _ = chi2_contingency(contingency_table)
            results.append(
                {
                    "column": column,
                    "statistic": statistic,
                    "p_value": p_value,
                    "independent": p_value > alpha,
                    "less_than_5_any_cell": less_than_5,
                }
            )
        return pd.DataFrame(results)

    def plot_qqplot(self, columns: list[str], figsize=(12, 12)) -> None:
        """
        Plot the QQ plot for the specified columns.
        The comparison is made against a standard normal distribution.

        Parameters
        ----------
        columns : list[str]
            Columns to plot.
        figsize : tuple, optional
            Figure size, by default (12, 12).
        """
        grid_shape = best_grid_shape(len(columns))

        fig, axs = plt.subplots(grid_shape[0], grid_shape[1], figsize=figsize)

        for ax, column in zip(axs.flat, columns):
            sm.qqplot(self.dataframe[column], line="s", ax=ax, fit=True)
            ax.set_title(f"{column}", fontsize="medium")
            ax.set_xlabel("Theoretical", fontsize="small")
            ax.set_ylabel("Observed", fontsize="small")

        plt.tight_layout()
        plt.show()
