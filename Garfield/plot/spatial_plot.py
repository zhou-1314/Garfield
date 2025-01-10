## modified by cell2location and scTM
## please refer to https://github.com/JinmiaoChenLab/scTM/blob/main/sctm/pl.py

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt

import textwrap
import numpy as np
import scanpy as sc

# import seaborn as sns
from matplotlib import rcParams
from matplotlib.axes import Axes
from collections import defaultdict
import pandas as pd
import anndata

# import matplotlib.pyplot as plt
# import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

# import pandas as pd
from matplotlib.patches import Patch

# from upsetplot import plot, from_contents
# from itertools import chain
from scanpy._utils import Empty, _empty
from scanpy.pl._tools.scatterplots import (
    _check_crop_coord,
    _check_img,
    _check_na_color,
    _check_scale_factor,
    _check_spatial_data,
    _check_spot_size,
)


def get_rgb_function(cmap, min_value, max_value):
    r"""Generate a function to map continous values to RGB values using colormap
    between min_value & max_value."""

    if min_value > max_value:
        raise ValueError("Max_value should be greater or than min_value.")

        # if min_value == max_value:
        #     warnings.warn(
        #         "Max_color is equal to min_color. It might be because of the data or
        #  bad
        #         parameter choice. "
        #         "If you are using plot_contours function try increasing
        # max_color_quantile
        #         parameter and"
        #         "removing cell types with all zero values."
        #     )

        def func_equal(x):
            factor = 0 if max_value == 0 else 0.5
            return cmap(np.ones_like(x) * factor)

        return func_equal

    def func(x):
        return cmap(
            (np.clip(x, min_value, max_value) - min_value) / (max_value - min_value)
        )

    return func


def rgb_to_ryb(rgb):
    """
    Converts colours from RGB colorspace to RYB

    Parameters
    ----------

    rgb
        numpy array Nx3

    Returns
    -------
    Numpy array Nx3
    """
    rgb = np.array(rgb)
    if len(rgb.shape) == 1:
        rgb = rgb[np.newaxis, :]

    white = rgb.min(axis=1)
    black = (1 - rgb).min(axis=1)
    rgb = rgb - white[:, np.newaxis]

    yellow = rgb[:, :2].min(axis=1)
    ryb = np.zeros_like(rgb)
    ryb[:, 0] = rgb[:, 0] - yellow
    ryb[:, 1] = (yellow + rgb[:, 1]) / 2
    ryb[:, 2] = (rgb[:, 2] + rgb[:, 1] - yellow) / 2

    mask = ~(ryb == 0).all(axis=1)
    if mask.any():
        norm = ryb[mask].max(axis=1) / rgb[mask].max(axis=1)
        ryb[mask] = ryb[mask] / norm[:, np.newaxis]

    return ryb + black[:, np.newaxis]


def ryb_to_rgb(ryb):
    """
    Converts colours from RYB colorspace to RGB

    Parameters
    ----------

    ryb
        numpy array Nx3

    Returns
    -------
    Numpy array Nx3
    """
    ryb = np.array(ryb)
    if len(ryb.shape) == 1:
        ryb = ryb[np.newaxis, :]

    black = ryb.min(axis=1)
    white = (1 - ryb).min(axis=1)
    ryb = ryb - black[:, np.newaxis]

    green = ryb[:, 1:].min(axis=1)
    rgb = np.zeros_like(ryb)
    rgb[:, 0] = ryb[:, 0] + ryb[:, 1] - green
    rgb[:, 1] = green + ryb[:, 1]
    rgb[:, 2] = (ryb[:, 2] - green) * 2

    mask = ~(ryb == 0).all(axis=1)
    if mask.any():
        norm = rgb[mask].max(axis=1) / ryb[mask].max(axis=1)
        rgb[mask] = rgb[mask] / norm[:, np.newaxis]

    return rgb + white[:, np.newaxis]


def plot_spatial_general(
    value_df,
    coords,
    labels,
    text=None,
    circle_radius=None,
    display_zeros=False,
    figsize=(10, 10),
    alpha_scaling=1.0,
    max_col=(np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
    max_color_quantile=0.98,
    show_img=True,
    img=None,
    img_alpha=1.0,
    adjust_text=False,
    plt_axis="off",
    axis_y_flipped=False,
    x_y_labels=("", ""),
    crop_x=None,
    crop_y=None,
    text_box_alpha=0.9,
    reorder_cmap=range(7),
    style="fast",
    colorbar_position="right",
    colorbar_label_kw={},
    colorbar_shape={},
    colorbar_tick_size=12,
    colorbar_grid=None,
    image_cmap="Greys_r",
    white_spacing=20,
):
    if value_df.shape[1] > 7:
        raise ValueError(
            "Maximum of 7 cell types / factors can be plotted at the moment"
        )

    def create_colormap(R, G, B):
        spacing = int(white_spacing * 2.55)

        N = 255
        M = 3

        alphas = np.concatenate(
            [[0] * spacing * M, np.linspace(0, 1.0, (N - spacing) * M)]
        )

        vals = np.ones((N * M, 4))
        #         vals[:, 0] = np.linspace(1, R / 255, N * M)
        #         vals[:, 1] = np.linspace(1, G / 255, N * M)
        #         vals[:, 2] = np.linspace(1, B / 255, N * M)
        for i, color in enumerate([R, G, B]):
            vals[:, i] = color / 255
        vals[:, 3] = alphas

        return ListedColormap(vals)

    # Create linearly scaled colormaps
    YellowCM = create_colormap(
        240, 228, 66
    )  # #F0E442 ['#F0E442', '#D55E00', '#56B4E9',
    # '#009E73', '#5A14A5', '#C8C8C8', '#323232']
    RedCM = create_colormap(213, 94, 0)  # #D55E00
    BlueCM = create_colormap(86, 180, 233)  # #56B4E9
    GreenCM = create_colormap(0, 158, 115)  # #009E73
    PinkCM = create_colormap(255, 105, 180)  # #C8C8C8
    WhiteCM = create_colormap(50, 50, 50)  # #323232
    PurpleCM = create_colormap(90, 20, 165)  # #5A14A5
    # LightGreyCM = create_colormap(240, 240, 240)  # Very Light Grey: #F0F0F0

    cmaps = [YellowCM, RedCM, BlueCM, GreenCM, PurpleCM, PinkCM, WhiteCM]

    cmaps = [cmaps[i] for i in reorder_cmap]

    with mpl.style.context(style):
        fig = plt.figure(figsize=figsize)
        if colorbar_position == "right":
            if colorbar_grid is None:
                colorbar_grid = (len(labels), 1)

            shape = {
                "vertical_gaps": 1.5,
                "horizontal_gaps": 0,
                "width": 0.15,
                "height": 0.2,
            }
            shape = {**shape, **colorbar_shape}

            gs = GridSpec(
                nrows=colorbar_grid[0] + 2,
                ncols=colorbar_grid[1] + 1,
                width_ratios=[1, *[shape["width"]] * colorbar_grid[1]],
                height_ratios=[1, *[shape["height"]] * colorbar_grid[0], 1],
                hspace=shape["vertical_gaps"],
                wspace=shape["horizontal_gaps"],
            )
            ax = fig.add_subplot(gs[:, 0], aspect="equal", rasterized=True)

        if colorbar_position == "bottom":
            if colorbar_grid is None:
                if len(labels) <= 3:
                    colorbar_grid = (1, len(labels))
                else:
                    n_rows = round(len(labels) / 3 + 0.5 - 1e-9)
                    colorbar_grid = (n_rows, 3)

            shape = {
                "vertical_gaps": 0.3,
                "horizontal_gaps": 0.6,
                "width": 0.2,
                "height": 0.035,
            }
            shape = {**shape, **colorbar_shape}

            gs = GridSpec(
                nrows=colorbar_grid[0] + 1,
                ncols=colorbar_grid[1] + 2,
                width_ratios=[0.3, *[shape["width"]] * colorbar_grid[1], 0.3],
                height_ratios=[1, *[shape["height"]] * colorbar_grid[0]],
                hspace=shape["vertical_gaps"],
                wspace=shape["horizontal_gaps"],
            )

            ax = fig.add_subplot(gs[0, :], aspect="equal", rasterized=True)

        if colorbar_position is None:
            ax = fig.add_subplot(aspect="equal", rasterized=True)

        if colorbar_position is not None:
            cbar_axes = []
            for row in range(1, colorbar_grid[0] + 1):
                for column in range(1, colorbar_grid[1] + 1):
                    cbar_axes.append(fig.add_subplot(gs[row, column]))

            n_excess = colorbar_grid[0] * colorbar_grid[1] - len(labels)
            if n_excess > 0:
                for i in range(1, n_excess + 1):
                    cbar_axes[-i].set_visible(False)

        ax.set_xlabel(x_y_labels[0])
        ax.set_ylabel(x_y_labels[1])

        if img is not None and show_img:
            ax.imshow(img, alpha=img_alpha, cmap=image_cmap)

        # crop images in needed
        if crop_x is not None:
            ax.set_xlim(crop_x[0], crop_x[1])
        if crop_y is not None:
            ax.set_ylim(crop_y[0], crop_y[1])

        if axis_y_flipped:
            ax.invert_yaxis()

        if plt_axis == "off":
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

        counts = value_df.values.copy()

        # plot spots as circles
        c_ord = list(np.arange(0, counts.shape[1]))
        colors = np.zeros((*counts.shape, 4))
        weights = np.zeros(counts.shape)

        for c in c_ord:
            min_color_intensity = counts[:, c].min()
            max_color_intensity = np.min(
                [np.quantile(counts[:, c], max_color_quantile), max_col[c]]
            )

            rgb_function = get_rgb_function(
                cmap=cmaps[c],
                min_value=min_color_intensity,
                max_value=max_color_intensity,
            )

            color = rgb_function(counts[:, c])
            color[:, 3] = color[:, 3] * alpha_scaling

            norm = mpl.colors.Normalize(
                vmin=min_color_intensity, vmax=max_color_intensity
            )

            if colorbar_position is not None:
                cbar_ticks = [
                    min_color_intensity,
                    np.mean([min_color_intensity, max_color_intensity]),
                    max_color_intensity,
                ]
                cbar_ticks = np.array(cbar_ticks)

                if max_color_intensity > 13:
                    cbar_ticks = cbar_ticks.astype(np.int32)
                else:
                    cbar_ticks = cbar_ticks.round(2)

                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmaps[c]),
                    cax=cbar_axes[c],
                    orientation="horizontal",
                    extend="both",
                    ticks=cbar_ticks,
                )

                cbar.ax.tick_params(labelsize=colorbar_tick_size)
                max_color = rgb_function(max_color_intensity / 1.5)
                cbar.ax.set_title(
                    labels[c],
                    **{
                        **{"size": 20, "color": max_color, "alpha": 1},
                        **colorbar_label_kw,
                    },
                )

            colors[:, c] = color
            weights[:, c] = np.clip(counts[:, c] / (max_color_intensity + 1e-10), 0, 1)
            weights[:, c][counts[:, c] < min_color_intensity] = 0

        colors_ryb = np.zeros((*weights.shape, 3))

        for i in range(colors.shape[0]):
            colors_ryb[i] = rgb_to_ryb(colors[i, :, :3])

        def kernel(w):
            return w**2

        kernel_weights = kernel(weights[:, :, np.newaxis])
        weighted_colors_ryb = (colors_ryb * kernel_weights).sum(
            axis=1
        ) / kernel_weights.sum(axis=1)
        weighted_colors = np.zeros((weights.shape[0], 4))
        weighted_colors[:, :3] = ryb_to_rgb(weighted_colors_ryb)
        weighted_colors[:, 3] = colors[:, :, 3].max(axis=1)

        if display_zeros:
            weighted_colors[weighted_colors[:, 3] == 0] = [
                210 / 255,
                210 / 255,
                210 / 255,
                1,
            ]

        ax.scatter(
            x=coords[:, 0], y=coords[:, 1], c=weighted_colors, s=circle_radius**2
        )

        # size in circles is radius
        # add text
        if text is not None:
            bbox_props = dict(boxstyle="round", ec="0.5", alpha=text_box_alpha, fc="w")
            texts = []
            for x, y, s in zip(
                np.array(text.iloc[:, 0].values).flatten(),
                np.array(text.iloc[:, 1].values).flatten(),
                text.iloc[:, 2].tolist(),
            ):
                texts.append(
                    ax.text(x, y, s, ha="center", va="bottom", bbox=bbox_props)
                )

            if adjust_text:
                from adjustText import adjust_text

                adjust_text(texts, arrowprops=dict(arrowstyle="->", color="w", lw=0.5))

    plt.grid(False)
    return fig


def plot_multi_patterns_spatial(
    adata,
    topic_prop,
    basis="spatial",
    bw=False,
    img=None,
    library_id=_empty,
    crop_coord=None,
    img_key=_empty,
    spot_size=None,
    na_color=None,
    scale_factor=None,
    scale_default=0.5,
    show_img=True,
    display_zeros=False,
    figsize=(10, 10),
    **kwargs,
):
    """Plot taken from cell2location at https://github.com/BayraktarLab/cell2location.
    Able to display zeros and also on umap through the basis function

    Args:
        adata (_type_): Adata object with spatial coordinates in adata.obsm['spatial']
        topic_prop (_type_): Topic proportion obtained from STAMP.
        basis (str, optional): Which basis to plot in adata.obsm. Defaults to "spatial".
        bw (bool, optional): Defaults to False.
        img (_type_, optional): . Defaults to None.
        library_id (_type_, optional): _description_. Defaults to _empty.
        crop_coord (_type_, optional): _description_. Defaults to None.
        img_key (_type_, optional): _description_. Defaults to _empty.
        spot_size (_type_, optional): _description_. Defaults to None.
        na_color (_type_, optional): _description_. Defaults to None.
        scale_factor (_type_, optional): _description_. Defaults to None.
        scale_default (float, optional): _description_. Defaults to 0.5.
        show_img (bool, optional): Whether to display spatial image. Sets to false
        automatically when displaying umap. Defaults to True.
        display_zeros (bool, optional): Whether to display cells that have low counts
        values to grey colour. Defaults to False.
        figsize (tuple, optional): Figsize of image. Defaults to (10, 10).

    Returns:
        _type_: Function taken from cell2location at
        https://cell2location.readthedocs.io/en/latest/_modules/cell2location/plt/plot_spatial.html#plot_spatial.
        Able to plot both on spatial and umap coordinates. Still very raw.
    """
    # get default image params if available
    library_id, spatial_data = _check_spatial_data(adata.uns, library_id)
    img, img_key = _check_img(spatial_data, img, img_key, bw=bw)
    spot_size = _check_spot_size(spatial_data, spot_size)
    scale_factor = _check_scale_factor(
        spatial_data, img_key=img_key, scale_factor=scale_factor
    )
    crop_coord = _check_crop_coord(crop_coord, scale_factor)
    na_color = _check_na_color(na_color, img=img)

    if scale_factor is not None:
        circle_radius = scale_factor * spot_size * 0.5 * scale_default
    else:
        circle_radius = spot_size * 0.5

    if show_img is True:
        kwargs["show_img"] = True
        kwargs["img"] = img

    kwargs["coords"] = adata.obsm[basis] * scale_factor

    fig = plot_spatial_general(
        value_df=topic_prop,
        labels=topic_prop.columns,
        circle_radius=circle_radius,
        figsize=figsize,
        display_zeros=display_zeros,
        **kwargs,
    )  # cell abundance values
    plt.gca().invert_yaxis()

    return fig


def plot_markers(
        adata: anndata.AnnData,
        groupby: str,
        mks: pd.DataFrame,
        n_genes: int = 5,
        kind: str = 'dotplot',
        remove_genes: list = [],
        **kwargs
):
    """
    Plot markers for specific groups.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data.
    groupby : str
        Column in `adata.obs` used for grouping cells.
    mks : DataFrame
        DataFrame containing marker statistics.
    n_genes : int, optional
        Number of top genes to plot per group. Default is 5.
    kind : str, optional
        Type of plot to create ('dotplot', 'violin', etc.). Default is 'dotplot'.
    remove_genes : list, optional
        List of genes to exclude from the plot. Default is an empty list.
    **kwargs : dict
        Additional keyword arguments passed to the plotting function.

    Returns
    -------
    matplotlib.Axes or None
        Axes object of the plot or None if plotting in place.
    """
    df = mks.reset_index()[['index', 'top_frac_group']].rename(columns={'index': 'gene',
                                                                        'top_frac_group': 'cluster'})
    var_tb = adata.raw.var if kwargs.get('use_raw', None) == True or adata.raw else adata.var
    remove_gene_set = set()
    for g_cat in remove_genes:
        if g_cat in var_tb.columns:
            remove_gene_set |= set(var_tb.index[var_tb[g_cat].values])
    df = df[~df.gene.isin(list(remove_gene_set))].copy()
    df1 = df.groupby('cluster').head(n_genes)
    mks_dict = defaultdict(list)
    for c, g in zip(df1.cluster, df1.gene):
        mks_dict[c].append(g)
    func = getattr(sc.pl, kind)
    if sc.__version__.startswith('1.4'):
        return func(adata, df1.gene.to_list(), groupby=groupby, **kwargs)
    else:
        return func(adata, mks_dict, groupby=groupby, **kwargs)


def niches_enrichment_barplot(
        enrichments,
        niche,
        type="enrichr",  # 默认为 enrichr 类型
        figsize=(10, 5),
        n_enrichments=5,
        qval_cutoff=0.05,
        title="auto",
):
    """
    Create a barplot for the enrichment results (either from Enrichr or GSEA).

    Parameters
    ----------
    enrichments : dict
        Dictionary of enrichment results where the key is the niche name and the value is a dataframe of enrichment results.
    niche : str
        The niche (cluster) to visualize enrichment for.
    type : str, optional
        The type of enrichment analysis ('enrichr' or 'gsea'). Default is 'gsea'.
    figsize : tuple, optional
        Figure size for the plot.
    n_enrichments : int, optional
        Number of top enrichment terms to display (default is 5).
    qval_cutoff : float, optional
        Adjusted p-value or FDR cutoff to filter enrichment terms (default is 0.05).
    title : str, optional
        Title of the plot (default is 'auto', which uses the first gene set's name).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axis containing the plot.
    """

    # Check if the niche exists in enrichments
    if niche not in enrichments:
        raise KeyError(f"Niche '{niche}' not found in enrichments.")

    # Extract enrichment data for the given niche
    enrichment = enrichments[niche]

    if type == "enrichr":
        # Filter by qval_cutoff and sort by Adjusted P-value
        enrichment = enrichment.loc[enrichment["Adjusted P-value"] < qval_cutoff, :]
        enrichment = enrichment.sort_values("Adjusted P-value")
        enrichment = enrichment.iloc[:n_enrichments, :]

        # Set title
        if title == "auto":
            title = enrichment["Gene_set"].iloc[0]

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(
            y=enrichment["Term"],
            width=-np.log(enrichment["Adjusted P-value"]),
            fill="blue",
            align="center",
        )

        # Format y-axis labels
        ax.set_yticklabels(
            [textwrap.fill(term, 24) for term in enrichment["Term"].values]
        )

        ax.set_xlabel("- Log Adjusted P-value")
        ax.set_title(title)
        ax.invert_yaxis()  # Reverse y-axis for top to bottom order

    elif type == "gsea":
        # Filter by qval_cutoff and sort by NES (Normalized Enrichment Score)
        enrichment = enrichment.loc[enrichment["NOM p-val"] < qval_cutoff, :]
        enrichment = enrichment[enrichment["NES"] > 0]
        enrichment = enrichment.sort_values("NES", ascending=False)
        enrichment["Term"] = enrichment["Term"].str.replace("_", " ")
        enrichment = enrichment.iloc[:n_enrichments, :]

        # Add -log q-value column for better visualization
        enrichment["-log_qval"] = -np.log(
            enrichment["FDR q-val"].astype("float") + 1e-7
        )

        # Set title
        if title == "auto":
            title = enrichment["Name"].iloc[0]

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(y=enrichment["Term"], width=enrichment["NES"], align="center")

        ax.set_xlabel("NES")
        ax.set_title(title)

        # Format y-axis labels
        ax.set_yticklabels(
            [textwrap.fill(term, 24) for term in enrichment["Term"].values]
        )

        ax.invert_yaxis()  # Reverse y-axis for top to bottom order

    else:
        raise ValueError("Unsupported enrichment type. Choose either 'enrichr' or 'gsea'.")

    plt.tight_layout()
    return ax


def niches_enrichment_dotplot(
    enrichments,
    niche,
    type="gsea",  # 默认为 gsea 类型
    figsize=(10, 5),
    n_enrichments=10,
    title="auto",
    cmap=None,
    qval_cutoff=0.05
):
    """
    Create a dotplot for the enrichment results (either from Enrichr or GSEA).

    Parameters
    ----------
    enrichments : dict
        Dictionary of enrichment results where the key is the niche name and the value is a dataframe of enrichment results.
    niche : str
        The niche (cluster) to visualize enrichment for.
    type : str, optional
        The type of enrichment analysis ('enrichr' or 'gsea'). Default is 'gsea'.
    figsize : tuple, optional
        Figure size for the plot.
    n_enrichments : int, optional
        Number of top enrichment terms to display (default is 10).
    title : str, optional
        Title of the plot (default is 'auto', which uses the first gene set's name).
    cmap : matplotlib.colors.Colormap, optional
        Colormap for the plot.
    qval_cutoff : float, optional
        Adjusted p-value or FDR cutoff to filter enrichment terms (default is 0.05).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axis containing the plot.
    """
    # Check if the niche exists in enrichments
    if niche not in enrichments:
        raise KeyError(f"Niche '{niche}' not found in enrichments.")

    # Extract enrichment data for the given niche
    enrichment = enrichments[niche]

    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    if type == "enrichr":
        # Process Enrichr results
        enrichment["gene_size"] = enrichment["Overlap"].str.split("/").str[1].astype(int)
        enrichment["-log_qval"] = -np.log(enrichment["Adjusted P-value"])
        enrichment["gene_ratio"] = enrichment["Overlap"].str.split("/").str[0].astype(int) / enrichment["gene_size"]

        # Filter by q-value cutoff
        enrichment = enrichment.loc[enrichment["Adjusted P-value"] < qval_cutoff, :]

        if enrichment.shape[0] < n_enrichments:
            n_enrichments = enrichment.shape[0]

        enrichment = enrichment.sort_values("gene_ratio", ascending=False)
        enrichment = enrichment.iloc[:n_enrichments, :]

        # Scatter plot
        scatter = ax.scatter(
            x=enrichment["gene_ratio"].values,
            y=enrichment["Term"].values,
            s=enrichment["gene_size"].values,
            c=enrichment["Combined Score"].values,
            cmap=cmap,
        )

        ax.set_xlabel("Gene Ratio")

        # Legends
        legend1 = ax.legend(
            *scatter.legend_elements(prop="sizes", num=5),
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
            title="Geneset Size",
            labelspacing=1,
            borderpad=1,
        )
        ax.legend(
            *scatter.legend_elements(prop="colors", num=5),
            bbox_to_anchor=(1.04, 0),
            loc="lower left",
            title="Combined Score",
            labelspacing=1,
            borderpad=1,
        )
        ax.add_artist(legend1)

        # Format y-axis labels
        ax.set_yticklabels([textwrap.fill(term, 24) for term in enrichment["Term"].values])

        # Set plot title
        if title == "auto":
            ax.set_title(enrichment["Gene_set"].values[0])

    elif type == "gsea":
        # Process GSEA results
        enrichment["gene_size"] = enrichment["Tag %"].str.split("/").str[1].astype(int)
        enrichment["-log_qval"] = -np.log(enrichment["FDR q-val"].astype(float) + 1e-7)
        enrichment["gene_ratio"] = enrichment["Tag %"].str.split("/").str[0].astype(int) / enrichment["gene_size"]

        # Filter by q-value cutoff
        enrichment = enrichment.loc[enrichment["FDR q-val"] < qval_cutoff, :]

        if enrichment.shape[0] < n_enrichments:
            n_enrichments = enrichment.shape[0]

        enrichment = enrichment.sort_values("-log_qval", ascending=False)
        enrichment = enrichment.iloc[:n_enrichments, :]

        # Scatter plot
        scatter = ax.scatter(
            x=enrichment["-log_qval"].values,
            y=enrichment["Term"].values,
            s=enrichment["gene_ratio"].values.astype(float),
            c=enrichment["NES"].values,
            cmap=cmap,
        )

        ax.set_xlabel("-log q_val")

        # Legends
        legend1 = ax.legend(
            *scatter.legend_elements(prop="sizes", num=5),
            bbox_to_anchor=(1, 1),
            loc="upper left",
            title="Gene Ratio",
            labelspacing=1,
            borderpad=1,
        )
        ax.legend(
            *scatter.legend_elements(prop="colors", num=5),
            bbox_to_anchor=(1, 0),
            loc="lower left",
            title="NES",
            labelspacing=1,
            borderpad=1,
        )
        ax.add_artist(legend1)

        # Format y-axis labels
        ax.set_yticklabels([textwrap.fill(term, 30) for term in enrichment["Term"].values])

        # Set plot title
        if title == "auto":
            ax.set_title(enrichment["Name"].values[0])

        ax.invert_yaxis()  # Reverse y-axis for top to bottom order

    else:
        raise ValueError("Unsupported enrichment type. Choose either 'enrichr' or 'gsea'.")

    # Tight layout for better visualization
    plt.tight_layout()
    return ax
