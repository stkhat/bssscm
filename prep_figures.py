"""
Generate some figures used in the paper
Usage:
    python3 prep_figures.py
"""

import json
import tempfile

import cv2
import fiona
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import seaborn as sns
from rasterio.features import rasterize, shapes

plt.rcParams["font.size"] = 15
START_YEAR_MB = 1985
EPSG = 5641


def polygonize_raster(in_raster, band=1, mask=None, epsg_out=4326):
    """Polygonize a raster image."""
    # Define a generator of GeoJSON features
    with rasterio.Env():
        with rasterio.open(in_raster) as src:
            image = src.read(band)
            epsg = src.crs.to_epsg()
            results = (
                {"properties": {"raster_val": v}, "geometry": s}
                for s, v in shapes(image, mask=mask, transform=src.transform)
            )

    # get geometries
    geoms = list(results)
    gdf_from_raster = gpd.GeoDataFrame.from_features(geoms, crs=f"EPSG:{epsg}")

    return gdf_from_raster.to_crs(epsg=epsg_out)


def clip_raster_by_polygons(path_raster_in, path_polygon, path_raster_out):
    """Clip a raster by a polygon.
    Input:
    - path_raster_in: a raster file to be clipped
    - path_polygon: mask polygon

    Output:
    - path_raster_out: a clipped raster
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        epsg = rasterio.open(path_raster_in).crs.to_epsg()
        shp_file = f"{tmp_dir}/tmp.shp"
        gpd.read_file(path_polygon).to_crs(epsg=epsg).to_file(
            shp_file, driver="ESRI Shapefile"
        )

        with fiona.open(shp_file) as shp:
            shapes = [feature["geometry"] for feature in shp]

    with rasterio.open(path_raster_in) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta

    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )

    with rasterio.open(path_raster_out, "w", **out_meta) as dest:
        dest.write(out_image)


def get_mask(poly, tif_img):
    """
    Get a mask of a polygon for a given tif image.
    Make sure the given polygon has the same epsg as tif.
    """
    with rasterio.open(tif_img) as src:
        transform_out = src.transform
        shape_out = src.shape
    mask = rasterize([(poly, 1)], transform=transform_out, out_shape=shape_out)

    return mask


# def plot_forest_transition(tif_forest, out_name, buffer_size, show_year):
#     with rasterio.open(tif_forest) as src:
#         forest_map = src.read()

#     # Get PA boundary mask
#     gdf_pa = gpd.read_file(vec_boundary_pa).to_crs(epsg=EPSG)
#     gdf_pa_buffer = gdf_pa.buffer(buffer_size)
#     gdf_pa_ring = gdf_pa_buffer.difference(gdf_pa)

#     pa_boundary_arr = get_mask(gdf_pa_ring.geometry[0], tif_forest)
#     pa_boundary_arr = pa_boundary_arr.astype(float)
#     pa_boundary_arr[pa_boundary_arr == 0] = np.nan

#     # year_show = 2000
#     for year_show in [2000, 2010, 2020]:
#         img = forest_map[year_show - START_YEAR_MB]
#         f, ax = plt.subplots()
#         plt.imshow(img, cmap="Greens", vmax=1.5, vmin=0)
#         plt.imshow(pa_boundary_arr, cmap="Reds", vmax=1, vmin=0, interpolation="none")
#         plt.axis("off")
#         plt.tight_layout()
#         if show_year:
#             plt.text(
#                 1,
#                 1,
#                 f"Year: {year_show}",
#                 horizontalalignment="right",
#                 verticalalignment="top",
#                 transform=ax.transAxes,
#                 fontsize=12,
#             )
#         plt.savefig(
#             f"figure/{out_name}_{year_show}.pdf", bbox_inches="tight", pad_inches=0
#         )
#         plt.close()


def plot_forest_transition(tif_forest, out_name, buffer_size):
    with rasterio.open(tif_forest) as src:
        forest_map = src.read()

    # Get PA boundary mask
    gdf_pa = gpd.read_file(vec_boundary_pa).to_crs(epsg=EPSG)
    gdf_pa_buffer = gdf_pa.buffer(buffer_size)
    gdf_pa_ring = gdf_pa_buffer.difference(gdf_pa)

    pa_boundary_arr = get_mask(gdf_pa_ring.geometry[0], tif_forest)
    pa_boundary_arr = pa_boundary_arr.astype(float)
    pa_boundary_arr[pa_boundary_arr == 0] = np.nan

    # year_show = 2000
    img_2000 = forest_map[2000 - START_YEAR_MB]
    img_2010 = forest_map[2010 - START_YEAR_MB]
    img_2020 = forest_map[2020 - START_YEAR_MB]
    diff_1 = img_2000 - img_2010
    diff_2 = img_2010 - img_2020

    out_img = 255 * np.ones((*img_2000.shape, 3), dtype=int)
    out_img[img_2000 == 1] = [34, 139, 34]  # forestgreen
    out_img[diff_1 == 1] = [255, 255, 0]  # yellow
    out_img[diff_2 == 1] = [255, 0, 0]  # yellow

    f, ax = plt.subplots()
    plt.imshow(out_img)
    plt.imshow(pa_boundary_arr, cmap="Blues", vmax=1, vmin=0, interpolation="none")
    yellow_patch = mpatches.Patch(
        color="yellow", label="Deforested are during 2000-2010"
    )
    red_patch = mpatches.Patch(color="red", label="Deforested area during 2010-2020")
    blue_patch = mpatches.Patch(
        edgecolor="blue", color=None, linewidth=2, label="Project area"
    )
    green_patch = mpatches.Patch(color="green", label="Forest area at 2020")
    plt.legend(
        handles=[blue_patch, yellow_patch, red_patch, green_patch],
        loc="lower right",
        fontsize=8,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"figure/{out_name}.pdf", bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_df_rates_car(in_df, year_start):
    df = 100 * pd.read_csv(in_df)  # show in percent scale
    n_rrd = len(df) - 1
    years = range(year_start, 2021)
    cols_show = [f"df_rate_{year}" for year in years]
    for i in range(n_rrd):
        plt.plot(
            years, df.loc[i + 1, cols_show].values, color="gray", linewidth=1, alpha=0.7
        )
    plt.plot(years, df.loc[0, cols_show].values, color="red")
    plt.ylabel("Annual Deforestation Rate [%]")
    plt.tight_layout()
    plt.ylim(0, 1.5)
    plt.savefig("figure/deforestation_rates_car.pdf")


def export_rrd_polygon_valparaiso(tif_acre_rrd):
    """
    Create a similar image as Figure 3.1 in the PDD.
    """

    with rasterio.open(tif_acre_rrd) as src:
        img_georeferenced = src.read([1, 2, 3]).transpose(1, 2, 0)  # Bands to RGB

    # Define the target color
    target_color = np.array([255, 211, 127])
    color_tolerance = 50
    lower_bound = target_color - color_tolerance
    upper_bound = target_color + color_tolerance

    # Mask the area of the target color
    mask_target_color = cv2.inRange(img_georeferenced, lower_bound, upper_bound)
    mask_target_color[700:, :400] = 0

    # Polygonize the mask
    with tempfile.TemporaryDirectory() as dir_tmp:
        meta = rasterio.open(tif_acre_rrd).meta
        meta.update({"count": 1, "nodata": 0})
        tif_tmp = f"{dir_tmp}/rrd_tmp.tif"
        with rasterio.open(tif_tmp, "w", **meta) as dst:
            dst.write_band(1, mask_target_color)

        poly = polygonize_raster(tif_tmp).query("raster_val==255")
        poly.to_file("figure/valparaiso_rrd.geojson", driver="GeoJSON")


def plot_deforestation_rates():
    area_rrd_pdd = 4651620
    area_pa_pdd = 28096
    baseline_pdd = pd.read_csv(
        "data/empirical/valparaiso_project_baseline.csv", header=None
    )  # From table 3.5 in the pdd
    baseline_pdd["rate_pa_pdd"] = baseline_pdd.iloc[:, 1:].sum(axis=1) / area_pa_pdd
    baseline_pdd.rename(columns={0: "year"}, inplace=True)
    rrd_def_pdd = pd.read_csv(
        "data/empirical/valparaiso_project_rrd.csv"
    )  # From table 3.2 in the pdd
    rrd_def_pdd["rate_rrd_pdd"] = rrd_def_pdd["def_ha"] / area_rrd_pdd

    with open("data/empirical/result_vcs_1113.json") as f:
        res = json.load(f)
    years = [*range(res["year_start"] + 1, res["year_end"] + 1)]
    rrd_calc = res["annual_deforestation_rate_rrd"]
    pa_calc = res["annual_deforestation_rate_pa_prj"]
    df_calc = pd.DataFrame(
        [years, rrd_calc, pa_calc], index=["year", "rate_rrd_mb", "rate_pa_mb"]
    ).T
    df_calc["year"] = df_calc["year"].astype(int)

    df_out = df_calc.merge(
        baseline_pdd[["year", "rate_pa_pdd"]], on="year", how="outer"
    ).merge(rrd_def_pdd[["year", "rate_rrd_pdd"]], on="year", how="outer")
    df_out = df_out.sort_values("year").reindex().query("2000<=year<=2020")
    df_out.iloc[:, 1:] = 100 * df_out.iloc[:, 1:]

    sns.lineplot(
        df_out,
        x="year",
        y="rate_pa_pdd",
        label="PA baseline (PDD)",
        linestyle="dashed",
        color="red",
    )
    sns.lineplot(
        df_out,
        x="year",
        y="rate_pa_mb",
        label="PA observed (MapBiomas)",
        linestyle="solid",
        color="red",
    )
    sns.lineplot(
        df_out,
        x="year",
        y="rate_rrd_mb",
        label="RRD observed (MapBiomas)",
        linestyle="solid",
        color="#377EB8",
    )
    plt.axvline(x=2011, color="gray", linestyle="dotted")
    plt.legend(fontsize=12)
    plt.ylabel("Annual deforestation rate [%]")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig("figure/valparaiso_trends.pdf")
    plt.close()

    df_out.to_csv("data/empirical/df_rate_merged.csv", index=None)


def compare_mapbiomas_tmf():
    boundary = "rrd"
    df = pd.read_csv("data/empirical/da_comparison_vcs_1113.csv")
    df.replace({"tmf": "TMF", "mapbiomas": "MapBiomas"}, inplace=True)
    df["year"] = df["year"].astype(int)
    sns.lineplot(
        df.query(f'boundary=="{boundary}" & dataset!="hansen"'),
        x="year",
        y="value",
        hue="dataset",
    )
    plt.xlabel("Year")
    plt.ylabel("Annual deforestation area [Ha]")
    plt.xticks([*range(2000, 2021, 5)])
    plt.tight_layout()
    plt.savefig(f"figure/compare_dataset_{boundary}.pdf")
    plt.close()


if __name__ == "__main__":
    tif_forest = "data/image/forest_map_mapbiomas.tif"
    tif_forest_clipped = "data/image/forest_map_mapbiomas_clipped.tif"
    vec_boundary_rec = "data/polygons/pa_around_rectangle.geojson"
    vec_boundary_pa = "data/polygons/pa_boundary_vcs_1113.geojson"
    tif_acre_rrd = "data/image/valparaiso_map_modified.tif"
    in_df = "data/empirical/df_long_mapbiomas.csv"

    # Do once
    clip_raster_by_polygons(tif_forest, vec_boundary_rec, tif_forest_clipped)

    # Forest map
    plot_forest_transition(tif_forest, "forestmap", buffer_size=1000, show_year=False)
    plot_forest_transition(tif_forest_clipped, "forestmap_clipped", buffer_size=500)

    export_rrd_polygon_valparaiso(tif_acre_rrd)
    plot_deforestation_rates()
    compare_mapbiomas_tmf()

    # Plot deforestation rates
    plot_df_rates_car(in_df, year_start=1995)
