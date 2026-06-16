from pathlib import Path

src = Path("doppler_sim/application.py").read_text(encoding="utf-8").splitlines()
block = src[1302:2060]
header = Path("doppler_sim/specg/_explorer_header.py").read_text(encoding="utf-8")
out_lines = []
skip = False
for line in block:
    if line.startswith("def specg_save_atoms"):
        skip = True
    if skip:
        if line.startswith("def specg_build_reassignment_extras"):
            skip = False
        else:
            continue
    if line.startswith("# ---"):
        continue
    if "Spectrogram Explorer" in line and line.startswith("#"):
        continue
    out_lines.append(line)

Path("doppler_sim/specg/explorer.py").write_text(header + "\n".join(out_lines), encoding="utf-8")
print("wrote", len(out_lines), "lines")
