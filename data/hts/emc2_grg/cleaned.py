"""Cleaning the GRG EMC²."""

import pandas as pd
import numpy as np
import data.hts.hts as hts

def configure(context):
    context.stage("data.hts.emc2_grg.raw")

PURPOSE_MAP = {
    "home": [1, 2],
    "work": [11, 12, 13, 81],
    "education": [21, 22, 23, 24, 25, 26, 27, 28, 29],
    "shop": [30, 31, 32, 33, 34, 35, 82],
    "leisure": [51, 52, 53, 54],
    "other": [41, 42, 43, 44, 45, 61, 62, 63, 64, 71, 72, 73, 74, 91]
}

MODES_MAP = {
    "car": [13, 15, 21, 81, 95],  # 95: tractors, agriculture machines
    "car_passenger": [14, 16, 22, 82],
    "pt": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 51, 52, 53, 61, 62, 71, 72, 73, 91, 92, 94],  # 61: Taxi, 62: VTC
    "bike": [10, 11, 12, 17, 18, 19, 93, 96],  # 96: scooter, solowheel...
    "walk": [1]
}

def zones_to_departement_ids(zones: pd.Series):
    assert not zones.isna().sum()
    departement_ids = pd.Series(index=zones.index, name="departement_id")
    # In EMC² GRG, every zone id that starts with a digit between 1 and 7 (included) is in the departement 38.
    departement_ids[zones.str[0].astype(int).between(1, 7)] = "38"

    # For each zone id starting with "8" or "9", the departement id is the 2 and 3rd digits of the zone id.
    departement_ids[zones.str[0].astype(int).between(8, 9)] = zones.str[1:3]

    assert not departement_ids.isna().sum()
    return departement_ids

def execute(context):
    df_households, df_persons, df_trips = context.stage("data.hts.emc2_grg.raw")

    # Clean zone identifiers
    df_households["MP2"] = df_households["MP2"].str[:-3]
    df_persons["PP2"] = df_persons["PP2"].str[:-3]
    df_trips["DP2"] = df_trips["DP2"].str[:-3]

    # Merge departement into households
    df_households["departement_id"] = zones_to_departement_ids(df_households["MP2"])

    # Transform original IDs to integer (they are hierarchichal)
    df_households["emc2_household_id"] = (df_households["ECH"] + df_households["MP2"]).astype(int)
    df_persons["emc2_person_id"] = df_persons["PER"].astype(np.int)
    df_persons["emc2_household_id"] = (df_persons["ECH"] + df_persons["PP2"]).astype(int)
    df_trips["emc2_person_id"] = df_trips["PER"].astype(np.int)
    df_trips["emc2_household_id"] = (df_trips["ECH"] + df_trips["DP2"]).astype(int)
    df_trips["emc2_trip_id"] = df_trips["NDEP"].astype(np.int)

    # Construct new IDs for households, persons and trips (which are unique globally)
    df_households["household_id"] = np.arange(len(df_households))

    df_persons = pd.merge(
        df_persons, df_households[["emc2_household_id", "household_id", "departement_id"]],
        on = ["emc2_household_id"]
    ).sort_values(by = ["household_id", "emc2_person_id"])
    df_persons["person_id"] = np.arange(len(df_persons))

    df_trips = pd.merge(
        df_trips, df_persons[["emc2_person_id", "emc2_household_id", "person_id", "household_id"]],
        on = ["emc2_person_id", "emc2_household_id"]
    ).sort_values(by = ["household_id", "person_id", "emc2_trip_id"])
    df_trips["trip_id"] = np.arange(len(df_trips))

    # Trip flags
    df_trips = hts.compute_first_last(df_trips)

    # Weight
    df_persons["person_weight"] = df_persons["COEP"].astype(np.float)
    df_households["household_weight"] = df_households["COEM"].astype(np.float)

    # Clean age
    df_persons["age"] = df_persons["P4"].astype(np.int)

    # Clean sex
    df_persons.loc[df_persons["P2"] == 1, "sex"] = "male"
    df_persons.loc[df_persons["P2"] == 2, "sex"] = "female"
    df_persons["sex"] = df_persons["sex"].astype("category")

    # Household size
    df_size = df_persons.groupby("household_id").size().reset_index(name = "household_size")
    df_households = pd.merge(df_households, df_size, on = "household_id")

    # Clean departement
    df_trips["origin_departement_id"] = zones_to_departement_ids(df_trips["D3"])
    df_trips["destination_departement_id"] = zones_to_departement_ids(df_trips["D7"])

    df_households["departement_id"] = df_households["departement_id"].astype("category")
    df_persons["departement_id"] = df_persons["departement_id"].astype("category")
    df_trips["origin_departement_id"] = df_trips["origin_departement_id"].astype("category")
    df_trips["destination_departement_id"] = df_trips["destination_departement_id"].astype("category")

    # Clean employment
    df_persons["employed"] = df_persons["P7"].isin(["1", "2"])

    # Studies
    df_persons["studies"] = df_persons["P7"].isin(["3", "4", "5"])

    # Number of vehicles
    df_households["number_of_vehicles"] = df_households["M14"] + df_households["M5"]
    df_households["number_of_vehicles"] = df_households["number_of_vehicles"].astype(np.int)
    df_households["number_of_bikes"] = df_households["M20"].astype(np.int)

    # License
    df_persons["has_license"] = df_persons["P5"] == "1"

    # Has subscription (not availabile in EDGT 44 but available in EMC² GRG)
    df_persons["has_pt_subscription"] = df_persons["P10"] >= 2

    # Trip purpose
    df_trips["following_purpose"] = "invalid"
    df_trips["preceding_purpose"] = "invalid"

    for purpose, values in PURPOSE_MAP.items():
        df_trips.loc[df_trips["D5A"].isin(values), "following_purpose"] = purpose
        df_trips.loc[df_trips["D2A"].isin(values), "preceding_purpose"] = purpose

    assert np.count_nonzero(df_trips["following_purpose"] == "invalid") == 0
    assert np.count_nonzero(df_trips["preceding_purpose"] == "invalid") == 0

    df_trips["following_purpose"] = df_trips["following_purpose"].astype("category")
    df_trips["preceding_purpose"] = df_trips["preceding_purpose"].astype("category")

    # Trip mode
    df_trips["mode"] = "invalid"

    for mode, values in MODES_MAP.items():
        df_trips.loc[df_trips["MODP"].isin(values), "mode"] = mode

    print(df_trips[df_trips["mode"] == "invalid"][["mode", "MODP"]])

    assert np.count_nonzero(df_trips["mode"] == "invalid") == 0
    df_trips["mode"] = df_trips["mode"].astype("category")

    # Further trip attributes
    df_trips["euclidean_distance"] = df_trips["DOIB"]
    df_trips["routed_distance"] = df_trips["DIST"]

    # Trip times
    df_trips["departure_time"] = 3600.0 * (df_trips["D4"] // 100)  # hour
    df_trips["departure_time"] += 60.0 * (df_trips["D4"] % 100)  # minute

    df_trips["arrival_time"] = 3600.0 * (df_trips["D8"] // 100)  # hour
    df_trips["arrival_time"] += 60.0 * (df_trips["D8"] % 100)  # minute

    df_trips = df_trips.sort_values(by = ["household_id", "person_id", "trip_id"])
    df_trips = hts.fix_trip_times(df_trips)

    # Durations
    df_trips["trip_duration"] = df_trips["arrival_time"] - df_trips["departure_time"]
    hts.compute_activity_duration(df_trips)

    # Add weight to trips
    df_trips = pd.merge(
        df_trips, df_persons[["person_id", "COEQ"]], on = "person_id", how = "left"
    ).rename(columns = { "COEQ": "trip_weight" })
    df_persons["trip_weight"] = df_persons["COEQ"]

    # Chain length
    df_count = df_trips[["person_id"]].groupby("person_id").size().reset_index(name = "number_of_trips")
    df_persons = pd.merge(df_persons, df_count, on = "person_id", how = "left")
    df_persons["number_of_trips"] = df_persons["number_of_trips"].fillna(-1).astype(int)

    # Passenger attribute
    df_persons["is_passenger"] = df_persons["person_id"].isin(
        df_trips[df_trips["mode"] == "car_passenger"]["person_id"].unique()
    )

    # Calculate consumption units
    hts.check_household_size(df_households, df_persons)
    df_households = pd.merge(df_households, hts.calculate_consumption_units(df_persons), on = "household_id")

    # Socioprofessional class
    df_persons["socioprofessional_class"] = df_persons["P9"].fillna(8).astype(int)
    df_persons.loc[df_persons["socioprofessional_class"] > 6, "socioprofessional_class"] = 8
    df_persons.loc[df_persons["P7"] == "7", "socioprofessional_class"] = 7

    # Check departure and arrival times
    assert np.count_nonzero(df_trips["departure_time"].isna()) == 0
    assert np.count_nonzero(df_trips["arrival_time"].isna()) == 0

    # Fix activity types (because of inconsistent EGT data and removing in the timing fixing step)
    hts.fix_activity_types(df_trips)

    return df_households, df_persons, df_trips
