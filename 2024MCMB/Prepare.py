# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# data = [
#     ["Commercial Satellite Imagery (new tasking/fast delivery)", 12, 800,   0,   0, 24],
#     ["Search & Rescue Drone (DJI Matrice 350 RTK + H20T)",      15,   8,   0, 0.7,  2],
#     ["Towed Pinger Locator (TPL-25 / similar)",               250, 120, 6000,  8, 12],
#     ["Deep-water AUV (Bluefin-21 / similar)",                 180,  12, 4500, 20, 24],
#     ["Small ROV (BlueROV2 / similar)",                          8,   1,  100,  6,  6],
#     ["Sonobuoys (a set / batch deployment)",                   50,  40,  300,  8,  6],
#     ["Side-scan Sonar (EdgeTech 4200-class)",                 120,  30, 2000, 10, 10],
#     ["Rescue Submersible / DSRV-class",                       2000,0.2,  600, 12, 48],
# ]
#
# df0 = pd.DataFrame(
#     data,
#     columns=["Equipment and Parameters", "Cost(kUSD)", "Coverage(km2/h)", "Max Depth(m)", "Task Cycle(h)", "Mobilization(h)"]
# )
#
# zone_map_multi = {
#     "Search & Rescue Drone (DJI Matrice 350 RTK + H20T)": ["Surface", "Shallow-water Zone"],
#     "Sonobuoys (a set / batch deployment)": ["Surface", "Shallow-water Zone"],
#     "Side-scan Sonar (EdgeTech 4200-class)": ["Shallow-water Zone", "Deep-water Zone"],
#     "Deep-water AUV (Bluefin-21 / similar)": ["Shallow-water Zone", "Deep-water Zone"],
#     "Small ROV (BlueROV2 / similar)": ["Shallow-water Zone"],
#     "Rescue Submersible / DSRV-class": ["Deep-water Zone"],
#     "Commercial Satellite Imagery (new tasking/fast delivery)": ["Surface"],
#     "Towed Pinger Locator (TPL-25 / similar)": ["Deep-water Zone"],
# }
#
# df0["Zone"] = df0["Equipment and Parameters"].map(zone_map_multi)
# df0 = df0.explode("Zone").reset_index(drop=True)
#
# default_failure = {
#     "Commercial Satellite Imagery (new tasking/fast delivery)": 0.5,
#     "Search & Rescue Drone (DJI Matrice 350 RTK + H20T)": 1.0,
#     "Towed Pinger Locator (TPL-25 / similar)": 0.5,
#     "Deep-water AUV (Bluefin-21 / similar)": 0.8,
#     "Small ROV (BlueROV2 / similar)": 1.2,
#     "Sonobuoys (a set / batch deployment)": 1.5,
#     "Side-scan Sonar (EdgeTech 4200-class)": 0.7,
#     "Rescue Submersible / DSRV-class": 1.0,
# }
# default_precision = {
#     "Commercial Satellite Imagery (new tasking/fast delivery)": 300,
#     "Search & Rescue Drone (DJI Matrice 350 RTK + H20T)": 20,
#     "Towed Pinger Locator (TPL-25 / similar)": 500,
#     "Deep-water AUV (Bluefin-21 / similar)": 100,
#     "Small ROV (BlueROV2 / similar)": 50,
#     "Sonobuoys (a set / batch deployment)": 300,
#     "Side-scan Sonar (EdgeTech 4200-class)": 10,
#     "Rescue Submersible / DSRV-class": 200,
# }
# default_accuracy = {
#     "Commercial Satellite Imagery (new tasking/fast delivery)": 90,
#     "Search & Rescue Drone (DJI Matrice 350 RTK + H20T)": 92,
#     "Towed Pinger Locator (TPL-25 / similar)": 95,
#     "Deep-water AUV (Bluefin-21 / similar)": 98,
#     "Small ROV (BlueROV2 / similar)": 99,
#     "Sonobuoys (a set / batch deployment)": 94,
#     "Side-scan Sonar (EdgeTech 4200-class)": 98,
#     "Rescue Submersible / DSRV-class": 96,
# }
# default_maint = {
#     "Commercial Satellite Imagery (new tasking/fast delivery)": 500,
#     "Search & Rescue Drone (DJI Matrice 350 RTK + H20T)": 200,
#     "Towed Pinger Locator (TPL-25 / similar)": 5000,
#     "Deep-water AUV (Bluefin-21 / similar)": 3000,
#     "Small ROV (BlueROV2 / similar)": 200,
#     "Sonobuoys (a set / batch deployment)": 0,
#     "Side-scan Sonar (EdgeTech 4200-class)": 1500,
#     "Rescue Submersible / DSRV-class": 20000,
# }
# default_oper = {
#     "Commercial Satellite Imagery (new tasking/fast delivery)": 2000,
#     "Search & Rescue Drone (DJI Matrice 350 RTK + H20T)": 50,
#     "Towed Pinger Locator (TPL-25 / similar)": 15000,
#     "Deep-water AUV (Bluefin-21 / similar)": 8000,
#     "Small ROV (BlueROV2 / similar)": 500,
#     "Sonobuoys (a set / batch deployment)": 2000,
#     "Side-scan Sonar (EdgeTech 4200-class)": 6000,
#     "Rescue Submersible / DSRV-class": 50000,
# }
#
# failure_by_zone = {
#     ("Side-scan Sonar (EdgeTech 4200-class)", "Shallow-water Zone"): 0.05,
#     ("Side-scan Sonar (EdgeTech 4200-class)", "Deep-water Zone"): 0.20,
#     ("Deep-water AUV (Bluefin-21 / similar)", "Shallow-water Zone"): 0.20,
#     ("Deep-water AUV (Bluefin-21 / similar)", "Deep-water Zone"): 1.00,
#     ("Small ROV (BlueROV2 / similar)", "Shallow-water Zone"): 0.10,
#     ("Rescue Submersible / DSRV-class", "Deep-water Zone"): 0.40,
#     ("Search & Rescue Drone (DJI Matrice 350 RTK + H20T)", "Surface"): 1.00,
#     ("Search & Rescue Drone (DJI Matrice 350 RTK + H20T)", "Shallow-water Zone"): 1.10,
#     ("Sonobuoys (a set / batch deployment)", "Surface"): 0.50,
#     ("Sonobuoys (a set / batch deployment)", "Shallow-water Zone"): 0.70,
# }
# precision_by_zone = {
#     ("Side-scan Sonar (EdgeTech 4200-class)", "Shallow-water Zone"): 10.2,
#     ("Side-scan Sonar (EdgeTech 4200-class)", "Deep-water Zone"): 40.8,
#     ("Deep-water AUV (Bluefin-21 / similar)", "Shallow-water Zone"): 5.0,
#     ("Deep-water AUV (Bluefin-21 / similar)", "Deep-water Zone"): 6.0,
#     ("Small ROV (BlueROV2 / similar)", "Shallow-water Zone"): 2.5,
#     ("Rescue Submersible / DSRV-class", "Deep-water Zone"): 4.5,
#     ("Search & Rescue Drone (DJI Matrice 350 RTK + H20T)", "Surface"): 7.8,
#     ("Search & Rescue Drone (DJI Matrice 350 RTK + H20T)", "Shallow-water Zone"): 12.0,
#     ("Sonobuoys (a set / batch deployment)", "Surface"): 11.6,
#     ("Sonobuoys (a set / batch deployment)", "Shallow-water Zone"): 15.0,
# }
# accuracy_by_zone = {
#     ("Side-scan Sonar (EdgeTech 4200-class)", "Shallow-water Zone"): 98.0,
#     ("Side-scan Sonar (EdgeTech 4200-class)", "Deep-water Zone"): 95.0,
#     ("Deep-water AUV (Bluefin-21 / similar)", "Shallow-water Zone"): 98.6,
#     ("Deep-water AUV (Bluefin-21 / similar)", "Deep-water Zone"): 96.0,
#     ("Small ROV (BlueROV2 / similar)", "Shallow-water Zone"): 99.4,
#     ("Rescue Submersible / DSRV-class", "Deep-water Zone"): 98.9,
#     ("Search & Rescue Drone (DJI Matrice 350 RTK + H20T)", "Surface"): 92.0,
#     ("Search & Rescue Drone (DJI Matrice 350 RTK + H20T)", "Shallow-water Zone"): 90.0,
#     ("Sonobuoys (a set / batch deployment)", "Surface"): 94.0,
#     ("Sonobuoys (a set / batch deployment)", "Shallow-water Zone"): 92.0,
# }
# maint_by_zone = {}
# oper_by_zone = {}
#
# range_by_zone = {
#     ("Side-scan Sonar (EdgeTech 4200-class)", "Shallow-water Zone"): 224.0,
#     ("Side-scan Sonar (EdgeTech 4200-class)", "Deep-water Zone"): 400.0,
#     ("Deep-water AUV (Bluefin-21 / similar)", "Shallow-water Zone"): 67.0,
#     ("Deep-water AUV (Bluefin-21 / similar)", "Deep-water Zone"): 65.0,
#     ("Small ROV (BlueROV2 / similar)", "Shallow-water Zone"): 30.0,
#     ("Rescue Submersible / DSRV-class", "Deep-water Zone"): 30.0,
#     ("Search & Rescue Drone (DJI Matrice 350 RTK + H20T)", "Surface"): 18.4e6,
#     ("Search & Rescue Drone (DJI Matrice 350 RTK + H20T)", "Shallow-water Zone"): 2.0e6,
#     ("Sonobuoys (a set / batch deployment)", "Surface"): 6.0e6,
#     ("Sonobuoys (a set / batch deployment)", "Shallow-water Zone"): 2.0e6,
# }
#
# def get_val(equip, zone, defaults, overrides):
#     return overrides.get((equip, zone), defaults[equip])
#
# def get_range(equip, zone, fallback):
#     return range_by_zone.get((equip, zone), fallback)
#
# df = pd.DataFrame({
#     "Zone": df0["Zone"],
#     "Equipment and Parameters": df0["Equipment and Parameters"],
# })
#
# df["Failure Rate (%)"] = [
#     get_val(e, z, default_failure, failure_by_zone)
#     for e, z in zip(df["Equipment and Parameters"], df["Zone"])
# ]
# df["Detection Range"] = [
#     get_range(e, z, float(r))
#     for e, z, r in zip(df["Equipment and Parameters"], df["Zone"], df0["Coverage(km2/h)"])
# ]
# df["Precision(cm)"] = [
#     get_val(e, z, default_precision, precision_by_zone)
#     for e, z in zip(df["Equipment and Parameters"], df["Zone"])
# ]
# df["Accuracy(%)"] = [
#     get_val(e, z, default_accuracy, accuracy_by_zone)
#     for e, z in zip(df["Equipment and Parameters"], df["Zone"])
# ]
# df["Price($)"] = (df0["Cost(kUSD)"] * 1000.0).astype(float)
# df["Maintenance Cost($)"] = [
#     get_val(e, z, default_maint, maint_by_zone)
#     for e, z in zip(df["Equipment and Parameters"], df["Zone"])
# ]
# df["Operating Cost($)"] = [
#     get_val(e, z, default_oper, oper_by_zone)
#     for e, z in zip(df["Equipment and Parameters"], df["Zone"])
# ]
#
# criteria = ["Failure Rate (%)", "Detection Range", "Precision(cm)", "Accuracy(%)", "Price($)", "Maintenance Cost($)", "Operating Cost($)"]
# benefit = np.array([False, True, False, True, False, False, False], dtype=bool)
# weights = np.array([0.10, 0.25, 0.10, 0.25, 0.15, 0.10, 0.05], dtype=float)
# weights = weights / weights.sum()
#
# def topsis(X, w, benefit_mask):
#     X = X.astype(float)
#     denom = np.sqrt((X ** 2).sum(axis=0))
#     denom[denom == 0] = 1.0
#     R = X / denom
#     V = R * w
#     A_plus = np.where(benefit_mask, V.max(axis=0), V.min(axis=0))
#     A_minus = np.where(benefit_mask, V.min(axis=0), V.max(axis=0))
#     D_plus = np.sqrt(((V - A_plus) ** 2).sum(axis=1))
#     D_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))
#     return D_minus / (D_plus + D_minus + 1e-12)
#
# candidates_by_zone = {
#     "Surface": [
#         "Search & Rescue Drone (DJI Matrice 350 RTK + H20T)",
#         "Sonobuoys (a set / batch deployment)",
#     ],
#     "Shallow-water Zone": [
#         "Side-scan Sonar (EdgeTech 4200-class)",
#         "Small ROV (BlueROV2 / similar)",
#         "Deep-water AUV (Bluefin-21 / similar)",
#         "Search & Rescue Drone (DJI Matrice 350 RTK + H20T)",
#         "Sonobuoys (a set / batch deployment)",
#     ],
#     "Deep-water Zone": [
#         "Side-scan Sonar (EdgeTech 4200-class)",
#         "Rescue Submersible / DSRV-class",
#         "Deep-water AUV (Bluefin-21 / similar)",
#     ],
# }
#
# results = []
# for zone, g in df.groupby("Zone", sort=False):
#     cand = candidates_by_zone.get(zone, [])
#     gg = g[g["Equipment and Parameters"].isin(cand)].copy()
#     X = gg[criteria].to_numpy(float)
#     C = topsis(X, weights, benefit)
#     tmp = gg[["Zone", "Equipment and Parameters"]].copy()
#     tmp["Closeness (C)"] = C
#     tmp["Rank"] = (-tmp["Closeness (C)"]).rank(method="min").astype(int)
#     results.append(tmp.sort_values("Rank"))
#
# out = pd.concat(results, ignore_index=True)
#
# type_zone_equipment = [
#     ("SSS", "Shallow-water Zone", "Side-scan Sonar (EdgeTech 4200-class)"),
#     ("SSS", "Deep-water Zone", "Side-scan Sonar (EdgeTech 4200-class)"),
#
#     ("SRD", "Surface", "Search & Rescue Drone (DJI Matrice 350 RTK + H20T)"),
#     ("SRD", "Shallow-water Zone", "Search & Rescue Drone (DJI Matrice 350 RTK + H20T)"),
#
#     ("AUR", "Shallow-water Zone", "Small ROV (BlueROV2 / similar)"),
#     ("AUR", "Deep-water Zone", "Rescue Submersible / DSRV-class"),
#
#     ("MSRB", "Surface", "Sonobuoys (a set / batch deployment)"),
#     ("MSRB", "Shallow-water Zone", "Sonobuoys (a set / batch deployment)"),
#
#     ("AUV", "Shallow-water Zone", "Deep-water AUV (Bluefin-21 / similar)"),
#     ("AUV", "Deep-water Zone", "Deep-water AUV (Bluefin-21 / similar)"),
# ]
#
# plot_key = pd.DataFrame(type_zone_equipment, columns=["Type", "Zone", "Equipment and Parameters"])
# plot_df = plot_key.merge(out, on=["Zone", "Equipment and Parameters"], how="left")
#
# plot_df["Score"] = 3.0 * plot_df["Closeness (C)"]
#
# zone_order = ["Surface", "Shallow-water Zone", "Deep-water Zone"]
# zone_label = {
#     "Surface": "Surface Water Area",
#     "Shallow-water Zone": "shallow water area",
#     "Deep-water Zone": "deep water area",
# }
# type_order = ["SSS", "SRD", "AUR", "MSRB", "AUV"]
#
# pivot = (
#     plot_df.pivot(index="Type", columns="Zone", values="Score")
#     .reindex(index=type_order)
#     .reindex(columns=zone_order)
# )
#
# x = np.arange(len(type_order))
# w = 0.25
#
# fig, ax = plt.subplots(figsize=(10, 5))
#
# colors = {"Surface": "red", "Shallow-water Zone": "black", "Deep-water Zone": "#2aa879"}
#
# for i, z in enumerate(zone_order):
#     y = pivot[z].to_numpy(float)
#     m = ~np.isnan(y)
#     ax.bar(
#         x[m] + (i - 1) * w,
#         y[m],
#         width=w,
#         label=zone_label[z],
#         color=colors[z],
#         edgecolor="black"
#     )
#
# ax.set_xticks(x)
# ax.set_xticklabels(type_order)
# ax.set_ylabel("Scores")
# ax.set_ylim(0, 3.0)
# ax.legend(loc="upper right", frameon=True)
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# plt.tight_layout()
# plt.show()
#
# print(out.to_string(index=False))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = [
    ["Commercial Satellite Imagery (new tasking/fast delivery)", 12, 800,   0,   0, 24],
    ["Search & Rescue Drone (DJI Matrice 350 RTK + H20T)",      15,   8,   0, 0.7,  2],
    ["Towed Pinger Locator (TPL-25 / similar)",               250, 120, 6000,  8, 12],
    ["Deep-water AUV (Bluefin-21 / similar)",                 180,  12, 4500, 20, 24],
    ["Small ROV (BlueROV2 / similar)",                          8,   1,  100,  6,  6],
    ["Sonobuoys (a set / batch deployment)",                   50,  40,  300,  8,  6],
    ["Side-scan Sonar (EdgeTech 4200-class)",                 120,  30, 2000, 10, 10],
    ["Rescue Submersible / DSRV-class",                       2000,0.2,  600, 12, 48],
]

df0 = pd.DataFrame(
    data,
    columns=["Equipment and Parameters", "Cost(kUSD)", "Coverage(km2/h)", "Max Depth(m)", "Task Cycle(h)", "Mobilization(h)"]
)

zone_map_multi = {
    "Commercial Satellite Imagery (new tasking/fast delivery)": ["Surface"],
    "Search & Rescue Drone (DJI Matrice 350 RTK + H20T)": ["Surface", "Shallow-water Zone"],
    "Sonobuoys (a set / batch deployment)": ["Surface", "Shallow-water Zone"],
    "Small ROV (BlueROV2 / similar)": ["Shallow-water Zone"],
    "Deep-water AUV (Bluefin-21 / similar)": ["Shallow-water Zone", "Deep-water Zone"],
    "Side-scan Sonar (EdgeTech 4200-class)": ["Shallow-water Zone", "Deep-water Zone"],
    "Towed Pinger Locator (TPL-25 / similar)": ["Deep-water Zone"],
    "Rescue Submersible / DSRV-class": ["Deep-water Zone"],
}

df0["Zone"] = df0["Equipment and Parameters"].map(zone_map_multi)
df0 = df0.explode("Zone").reset_index(drop=True)

default_failure = {
    "Commercial Satellite Imagery (new tasking/fast delivery)": 0.5,
    "Search & Rescue Drone (DJI Matrice 350 RTK + H20T)": 1.0,
    "Towed Pinger Locator (TPL-25 / similar)": 0.5,
    "Deep-water AUV (Bluefin-21 / similar)": 0.8,
    "Small ROV (BlueROV2 / similar)": 1.2,
    "Sonobuoys (a set / batch deployment)": 1.5,
    "Side-scan Sonar (EdgeTech 4200-class)": 0.7,
    "Rescue Submersible / DSRV-class": 1.0,
}
default_precision = {
    "Commercial Satellite Imagery (new tasking/fast delivery)": 300,
    "Search & Rescue Drone (DJI Matrice 350 RTK + H20T)": 20,
    "Towed Pinger Locator (TPL-25 / similar)": 500,
    "Deep-water AUV (Bluefin-21 / similar)": 100,
    "Small ROV (BlueROV2 / similar)": 50,
    "Sonobuoys (a set / batch deployment)": 300,
    "Side-scan Sonar (EdgeTech 4200-class)": 10,
    "Rescue Submersible / DSRV-class": 200,
}
default_accuracy = {
    "Commercial Satellite Imagery (new tasking/fast delivery)": 90,
    "Search & Rescue Drone (DJI Matrice 350 RTK + H20T)": 92,
    "Towed Pinger Locator (TPL-25 / similar)": 95,
    "Deep-water AUV (Bluefin-21 / similar)": 98,
    "Small ROV (BlueROV2 / similar)": 99,
    "Sonobuoys (a set / batch deployment)": 94,
    "Side-scan Sonar (EdgeTech 4200-class)": 98,
    "Rescue Submersible / DSRV-class": 96,
}
default_maint = {
    "Commercial Satellite Imagery (new tasking/fast delivery)": 500,
    "Search & Rescue Drone (DJI Matrice 350 RTK + H20T)": 200,
    "Towed Pinger Locator (TPL-25 / similar)": 5000,
    "Deep-water AUV (Bluefin-21 / similar)": 3000,
    "Small ROV (BlueROV2 / similar)": 200,
    "Sonobuoys (a set / batch deployment)": 0,
    "Side-scan Sonar (EdgeTech 4200-class)": 1500,
    "Rescue Submersible / DSRV-class": 20000,
}
default_oper = {
    "Commercial Satellite Imagery (new tasking/fast delivery)": 2000,
    "Search & Rescue Drone (DJI Matrice 350 RTK + H20T)": 50,
    "Towed Pinger Locator (TPL-25 / similar)": 15000,
    "Deep-water AUV (Bluefin-21 / similar)": 8000,
    "Small ROV (BlueROV2 / similar)": 500,
    "Sonobuoys (a set / batch deployment)": 2000,
    "Side-scan Sonar (EdgeTech 4200-class)": 6000,
    "Rescue Submersible / DSRV-class": 50000,
}

failure_by_zone = {
    ("Side-scan Sonar (EdgeTech 4200-class)", "Shallow-water Zone"): 0.05,
    ("Side-scan Sonar (EdgeTech 4200-class)", "Deep-water Zone"): 0.20,
    ("Deep-water AUV (Bluefin-21 / similar)", "Shallow-water Zone"): 0.20,
    ("Deep-water AUV (Bluefin-21 / similar)", "Deep-water Zone"): 1.00,
    ("Small ROV (BlueROV2 / similar)", "Shallow-water Zone"): 0.10,
    ("Rescue Submersible / DSRV-class", "Deep-water Zone"): 0.40,
    ("Search & Rescue Drone (DJI Matrice 350 RTK + H20T)", "Surface"): 1.00,
    ("Search & Rescue Drone (DJI Matrice 350 RTK + H20T)", "Shallow-water Zone"): 1.10,
    ("Sonobuoys (a set / batch deployment)", "Surface"): 0.50,
    ("Sonobuoys (a set / batch deployment)", "Shallow-water Zone"): 0.70,
}
precision_by_zone = {
    ("Side-scan Sonar (EdgeTech 4200-class)", "Shallow-water Zone"): 10.2,
    ("Side-scan Sonar (EdgeTech 4200-class)", "Deep-water Zone"): 40.8,
    ("Deep-water AUV (Bluefin-21 / similar)", "Shallow-water Zone"): 5.0,
    ("Deep-water AUV (Bluefin-21 / similar)", "Deep-water Zone"): 6.0,
    ("Small ROV (BlueROV2 / similar)", "Shallow-water Zone"): 2.5,
    ("Rescue Submersible / DSRV-class", "Deep-water Zone"): 4.5,
    ("Search & Rescue Drone (DJI Matrice 350 RTK + H20T)", "Surface"): 7.8,
    ("Search & Rescue Drone (DJI Matrice 350 RTK + H20T)", "Shallow-water Zone"): 12.0,
    ("Sonobuoys (a set / batch deployment)", "Surface"): 11.6,
    ("Sonobuoys (a set / batch deployment)", "Shallow-water Zone"): 15.0,
}
accuracy_by_zone = {
    ("Side-scan Sonar (EdgeTech 4200-class)", "Shallow-water Zone"): 98.0,
    ("Side-scan Sonar (EdgeTech 4200-class)", "Deep-water Zone"): 95.0,
    ("Deep-water AUV (Bluefin-21 / similar)", "Shallow-water Zone"): 98.6,
    ("Deep-water AUV (Bluefin-21 / similar)", "Deep-water Zone"): 96.0,
    ("Small ROV (BlueROV2 / similar)", "Shallow-water Zone"): 99.4,
    ("Rescue Submersible / DSRV-class", "Deep-water Zone"): 98.9,
    ("Search & Rescue Drone (DJI Matrice 350 RTK + H20T)", "Surface"): 92.0,
    ("Search & Rescue Drone (DJI Matrice 350 RTK + H20T)", "Shallow-water Zone"): 90.0,
    ("Sonobuoys (a set / batch deployment)", "Surface"): 94.0,
    ("Sonobuoys (a set / batch deployment)", "Shallow-water Zone"): 92.0,
}

range_by_zone = {
    ("Side-scan Sonar (EdgeTech 4200-class)", "Shallow-water Zone"): 224.0,
    ("Side-scan Sonar (EdgeTech 4200-class)", "Deep-water Zone"): 400.0,
    ("Deep-water AUV (Bluefin-21 / similar)", "Shallow-water Zone"): 67.0,
    ("Deep-water AUV (Bluefin-21 / similar)", "Deep-water Zone"): 65.0,
    ("Small ROV (BlueROV2 / similar)", "Shallow-water Zone"): 30.0,
    ("Rescue Submersible / DSRV-class", "Deep-water Zone"): 30.0,
    ("Search & Rescue Drone (DJI Matrice 350 RTK + H20T)", "Surface"): 18.4e6,
    ("Search & Rescue Drone (DJI Matrice 350 RTK + H20T)", "Shallow-water Zone"): 2.0e6,
    ("Sonobuoys (a set / batch deployment)", "Surface"): 6.0e6,
    ("Sonobuoys (a set / batch deployment)", "Shallow-water Zone"): 2.0e6,
}

def get_val(equip, zone, defaults, overrides):
    return overrides.get((equip, zone), defaults[equip])

def get_range(equip, zone, fallback):
    return range_by_zone.get((equip, zone), float(fallback))

df = pd.DataFrame({
    "Zone": df0["Zone"],
    "Equipment and Parameters": df0["Equipment and Parameters"],
})

df["Failure Rate (%)"] = [get_val(e, z, default_failure, failure_by_zone) for e, z in zip(df["Equipment and Parameters"], df["Zone"])]
df["Detection Range"] = [get_range(e, z, r) for e, z, r in zip(df["Equipment and Parameters"], df["Zone"], df0["Coverage(km2/h)"])]
df["Precision(cm)"] = [get_val(e, z, default_precision, precision_by_zone) for e, z in zip(df["Equipment and Parameters"], df["Zone"])]
df["Accuracy(%)"] = [get_val(e, z, default_accuracy, accuracy_by_zone) for e, z in zip(df["Equipment and Parameters"], df["Zone"])]
df["Price($)"] = (df0["Cost(kUSD)"] * 1000.0).astype(float)
df["Maintenance Cost($)"] = [get_val(e, z, default_maint, {}) for e, z in zip(df["Equipment and Parameters"], df["Zone"])]
df["Operating Cost($)"] = [get_val(e, z, default_oper, {}) for e, z in zip(df["Equipment and Parameters"], df["Zone"])]

criteria = ["Failure Rate (%)", "Detection Range", "Precision(cm)", "Accuracy(%)", "Price($)", "Maintenance Cost($)", "Operating Cost($)"]
benefit = np.array([False, True, False, True, False, False, False], dtype=bool)
weights = np.array([0.10, 0.25, 0.10, 0.25, 0.15, 0.10, 0.05], dtype=float)
weights = weights / weights.sum()

def topsis(X, w, benefit_mask):
    X = X.astype(float)
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    R = X / denom
    V = R * w
    A_plus = np.where(benefit_mask, V.max(axis=0), V.min(axis=0))
    A_minus = np.where(benefit_mask, V.min(axis=0), V.max(axis=0))
    D_plus = np.sqrt(((V - A_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))
    C = D_minus / (D_plus + D_minus + 1e-12)
    return C

zone_order = ["Surface", "Shallow-water Zone", "Deep-water Zone"]
df["Zone"] = pd.Categorical(df["Zone"], categories=zone_order, ordered=True)

out_list = []
for zone, g in df.groupby("Zone", sort=False, observed=False):
    X = g[criteria].to_numpy(float)
    C = topsis(X, weights, benefit)
    tmp = g[["Zone", "Equipment and Parameters"]].copy()
    tmp["Closeness (C)"] = C
    tmp["Rank"] = (-tmp["Closeness (C)"]).rank(method="min").astype(int)
    out_list.append(tmp)

out = pd.concat(out_list, ignore_index=True).sort_values(["Zone", "Rank"])
print(out.to_string(index=False))
print("--------------------------------------------------------------------------------------------------------------------")
table_cols = ["Zone", "Equipment and Parameters", "Failure Rate (%)", "Detection Range", "Precision(cm)", "Accuracy(%)", "Price($)", "Maintenance Cost($)", "Operating Cost($)"]
print(df[table_cols].sort_values(["Zone", "Equipment and Parameters"]).to_string(index=False))

short_name = {
    "Commercial Satellite Imagery (new tasking/fast delivery)": "SAT",
    "Search & Rescue Drone (DJI Matrice 350 RTK + H20T)": "SRD",
    "Sonobuoys (a set / batch deployment)": "SBUOY",
    "Small ROV (BlueROV2 / similar)": "ROV",
    "Deep-water AUV (Bluefin-21 / similar)": "AUV",
    "Side-scan Sonar (EdgeTech 4200-class)": "SSS",
    "Towed Pinger Locator (TPL-25 / similar)": "TPL",
    "Rescue Submersible / DSRV-class": "DSRV",
}

plot_df = out.copy()
plot_df["Type"] = plot_df["Equipment and Parameters"].map(short_name).fillna(plot_df["Equipment and Parameters"])

plot_df["Score"] = 3.0 * plot_df["Closeness (C)"]
plot_df["Score"] = plot_df["Score"].clip(0, 3)

cnt = plot_df.groupby("Type", observed=False)["Zone"].nunique()
keep_types = cnt[cnt >= 2].index
plot_df = plot_df[plot_df["Type"].isin(keep_types)].copy()

pivot = plot_df.pivot_table(index="Type", columns="Zone", values="Score", aggfunc="mean", observed=False)
pivot = pivot.reindex(columns=zone_order)

types = pivot.index.tolist()
x = np.arange(len(types))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 4.5))

offsets = {
    "Surface": -width,
    "Shallow-water Zone": 0.0,
    "Deep-water Zone": width,
}

for z in zone_order:
    y = pivot[z].to_numpy()
    mask = ~np.isnan(y)
    ax.bar(x[mask] + offsets[z], y[mask], width, label=z)

ax.set_xticks(x)
ax.set_xticklabels(types)
ax.set_ylabel("Scores")
ax.set_ylim(0, 3.0)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("topsis_scores_by_zone.png", dpi=300)
plt.show()
