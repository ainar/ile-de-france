"""
Loads the raw data of the specified HTS.

emc2_grg = EMC² Grande Région Grenobloise.

Adapted from the first implementation by Valentin Le Besond (IFSTTAR Nantes)
"""

import pandas as pd
import os

from .format import HOUSEHOLD_FORMAT, PERSON_FORMAT, TRIP_FORMAT

FILES = {
    "menage": "08a_EMC2_GRG_2020_MENAGE_Coef_300421.txt",
    "perso": "08b_EMC2_GRG_2020_PERSO_Coef_300421.txt",
    "depla": "08c_EMC2_GRG_2020_DEPLA_Dist2010_150421.txt",
}

HTS_FOLDER = "emc2_grg_2020"

def configure(context):
    context.config("data_path")


HOUSEHOLD_COLUMNS = {
    "MP2": str, "ECH": str, "COEM": float,
    "M14": int, "M20": int, "M5": int
}

PERSON_COLUMNS = {
    "ECH": str, "PER": int, "PP2": str,
    "P3": int, "P2": int, "P4": int,
    "P7": str, "P12": str,
    "P9": str, "P5": str, "P10": float,
    "COEP": float, "COEQ": float
}

TRIP_COLUMNS = {
    "ECH": str, "PER": int, "NDEP": int, "DP2": str,
    "D2A": int, "D5A": int, "D3": str, "D4": int,
    "D7": str, "D8": int,
    "D8C": int, "MODP": int, "DOIB": int, "DIST": int
}

def execute(context):
    # Load households
    df_household_dictionary = pd.DataFrame.from_records(
        HOUSEHOLD_FORMAT, columns = ["position", "size", "variable", "description"]
    )

    column_widths = df_household_dictionary["size"].values
    column_names = df_household_dictionary["variable"].values

    df_households = pd.read_fwf(
        os.path.join(context.config("data_path"), HTS_FOLDER, FILES["menage"]),
        widths = column_widths, header = None,
        names = column_names, usecols = list(HOUSEHOLD_COLUMNS.keys()), dtype = HOUSEHOLD_COLUMNS
    )

    # Load persons
    df_person_dictionary = pd.DataFrame.from_records(
        PERSON_FORMAT, columns = ["position", "size", "variable", "description"]
    )

    column_widths = df_person_dictionary["size"].values
    column_names = df_person_dictionary["variable"].values

    df_persons = pd.read_fwf(
        os.path.join(context.config("data_path"), HTS_FOLDER, FILES["perso"]),
        widths = column_widths, header = None,
        names = column_names, usecols = list(PERSON_COLUMNS.keys()), dtype = PERSON_COLUMNS
    )

    # Load trips
    df_trip_dictionary = pd.DataFrame.from_records(
        TRIP_FORMAT, columns = ["position", "size", "variable", "description"]
    )

    column_widths = df_trip_dictionary["size"].values
    column_names = df_trip_dictionary["variable"].values

    df_trips = pd.read_fwf(
        os.path.join(context.config("data_path"), HTS_FOLDER, FILES["depla"]),
        widths = column_widths, header = None,
        names = column_names, usecols = list(TRIP_COLUMNS.keys()), dtype = TRIP_COLUMNS
    )

    return df_households, df_persons, df_trips

def validate(context):
    for name in FILES:
        if not os.path.exists(os.path.join(context.config("data_path"), HTS_FOLDER, FILES[name])):
            raise RuntimeError("File missing from HTS: %s" % name)

    return [
        os.path.getsize(os.path.join(context.config("data_path"), HTS_FOLDER, FILES[name]))
        for name in FILES
    ]
