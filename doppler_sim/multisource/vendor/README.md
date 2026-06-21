# Vendored multisource dependencies

Self-contained copies of the Doppler 4.0 / 5.0 modules required by the
**Multisource Pass-By** tab only. Pass-By Sim, batch generation, and other
tabs do not import from here.

## Layout

```
vendor/
  doppler_4dot0/
    audio/          emission q(t′) extraction helpers
    physics/        3D kinematics + retarded-time propagation
    inverse/        band shares, body combine rules
    synthesis.py    _remove_dc, _render_path only
    config.py
  doppler_5dot0/
    sources/        14-source catalog, tonal extract, broadband beds
    inverse/        VS13 inverse synthesis + STFT fit
    physics/        tonal/broadband combine rules
    config.py
```

Loaded via `doppler_sim.multisource.bootstrap.ensure_vendor_on_path()`.
