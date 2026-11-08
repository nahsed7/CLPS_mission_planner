# Input schemas (loose — column names are matched by alias)

## Sites CSV
Required: site id, latitude, longitude  
Optional: name, elevation (m)

Accepted aliases include: `site_id|id|site`, `lat|latitude|latitude_deg`, `lon|longitude|longitude_deg`, `elevation_m|observer_elevation_m`.

## Horizon CSV
Required: `azimuth_deg`, horizon elevation  
Optional: `site_id` (for combined multi-site files)

Aliases: `horizon_elevation_deg|horizon_elev_deg|horizon_deg`.

## Geometry CSV (precomputed ephemeris)
Required: time, sun alt/az, earth alt/az  
Optional: `earth_radius` / `earth_angular_radius_deg`, `site_id`, precomputed `sunlit` / `earth_visible`

## Terrain product
Pair of files from datakit: `<id>_terrain.npz` + `<id>_terrain.json`  
Or a Model-3 style summary CSV with a `site_id` column.

## datakit NPZ
`ephemeris_*.npz` from `tools/datakit.py spice` with `utc`, `sun_km`, `earth_km` arrays.
