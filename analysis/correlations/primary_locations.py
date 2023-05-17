import pandas as pd
import geopandas as gpd
import scipy.spatial.distance
import scipy.stats
import numpy as np

import synthesis.population.spatial.primary.locations
import synthesis.population.spatial.home.locations
import synthesis.population.sampled


def configure(context):
    context.stage(
        synthesis.population.spatial.primary.locations,
        alias="primary_locations",
    )

    context.stage(
        synthesis.population.spatial.home.locations, alias="home_locations"
    )

    context.stage(synthesis.population.sampled, alias="persons")


def execute(context):
    work_locations, edu_locations = context.stage("primary_locations")
    home_locations = context.stage("home_locations")
    persons = context.stage("persons")

    work_locations.rename(columns={"geometry": "work_geometry"}, inplace=True)
    home_locations.rename(columns={"geometry": "home_geometry"}, inplace=True)

    commutes = pd.merge(
        persons, home_locations, on="household_id", how="right"
    )
    commutes = pd.merge(commutes, work_locations, on="person_id", how="right")
    commutes = commutes[["home_geometry", "work_geometry"]]
    home_geometry = gpd.GeoSeries(commutes["home_geometry"])
    work_geometry = gpd.GeoSeries(commutes["work_geometry"])

    home_dists = scipy.spatial.distance.pdist(
        np.vstack([home_geometry.x.values, home_geometry.y.values]).T
    )
    work_dists = scipy.spatial.distance.pdist(
        np.vstack([work_geometry.x.values, work_geometry.y.values]).T
    )

    print("Pearson Correlation")
    print(scipy.stats.pearsonr(home_dists, work_dists))

    print("Spearman Correlation")
    print(scipy.stats.spearmanr(home_dists, work_dists))

    print("Spearman Correlation")
    print(scipy.stats.kendalltau(home_dists, work_dists))
