#Star Check-
stars_with_windows = []

for _, row in df.iterrows():
    kepid = int(row["kepid"])
    host  = str(row.get("hostname", "")).strip()

   
    kept = int(per_star_window_counts.get(kepid, 0))
    total_windows = kept

    if kept <= 0:
        continue

    # planets from Exoplanet Archive 
    planets = fetch_ephemerides_for_host(host)
    nplanets = len(planets)

    print(f"KIC {kepid} ({host}) | windows={total_windows} | planets={nplanets} | kept={kept}")
    stars_with_windows.append(kepid)

print(f"\nunique stars with windows: {len(stars_with_windows)}")
