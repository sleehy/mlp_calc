from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# 0. 파일 경로 설정
# -------------------------
band_yaml_path = "plot_archive/mattersim/band/inputs/band.yaml"
experiment_csv_path = "experiment_points.csv"
output_png_path = "plot_archive/mattersim/band/plots/band_plot_with_experiment.png"


# -------------------------
# 1. band.yaml 읽기
# -------------------------
with open(band_yaml_path, "r") as f:
    data = yaml.safe_load(f)

phonons = data["phonon"]
segment_nqpoint = data["segment_nqpoint"]
labels = data.get("labels", None)

x_calc = np.array([p["distance"] for p in phonons])
freqs = np.array([[b["frequency"] for b in p["band"]] for p in phonons])


# -------------------------
# 2. band.yaml에서 고대칭점 위치 추출
# -------------------------
idx = 0
hs_points = []

# 첫 시작점
if labels:
    hs_points.append((labels[0][0], idx, phonons[idx]["q-position"], phonons[idx]["distance"]))
else:
    hs_points.append(("start", idx, phonons[idx]["q-position"], phonons[idx]["distance"]))

# 각 segment의 끝점
for s, nq in enumerate(segment_nqpoint):
    end_idx = idx + nq - 1

    if labels:
        label = labels[s][1]
    else:
        label = f"segment_{s}_end"

    hs_points.append((label, end_idx, phonons[end_idx]["q-position"], phonons[end_idx]["distance"]))

    idx += nq

print("High-symmetry point distances")
print("-" * 80)
for label, i, q, d in hs_points:
    print(f"{label:12s} index={i:5d}  distance={d:.8f}  q={q}")


# -------------------------
# 3. label 이름 정리
#    GAMMA 중복은 G, G2로 구분
# -------------------------
def normalize_label(label):
    label = str(label)

    if label.upper() in ["GAMMA", "Γ", "G"]:
        return "G"
    return label


calc_hs = {}
gamma_count = 0

for label, i, q, d in hs_points:
    key = normalize_label(label)

    if key == "G":
        gamma_count += 1
        if gamma_count == 1:
            key = "G"
        else:
            key = f"G{gamma_count}"

    calc_hs[key] = d

print("\ncalc_hs automatically generated")
print("-" * 80)
for k, v in calc_hs.items():
    print(f'"{k}": {v:.8f},')


# -------------------------
# 4. 웹 그래프에서 WPD로 뽑은 고대칭점 위치
#    네가 직접 넣은 값
# -------------------------
web_hs = {
    "G": 0.000,
    "X": 0.3048762,
    "K": 0.4124713,
    "G2": 0.736312,
    "L": 1.000,
}

path = ["G", "X", "K", "G2", "L"]


# -------------------------
# 5. calc_hs에 필요한 label이 있는지 확인
# -------------------------
missing = [p for p in path if p not in calc_hs]
if missing:
    raise ValueError(
        f"calc_hs에 다음 고대칭점이 없습니다: {missing}\n"
        f"현재 calc_hs keys: {list(calc_hs.keys())}\n"
        f"band.yaml의 labels 또는 BAND_LABELS를 확인하세요."
    )


# -------------------------
# 6. calc x좌표를 web x좌표로 변환
# -------------------------
def map_calc_to_web_x(x):
    eps = 1e-10

    for a, b in zip(path[:-1], path[1:]):
        x0_calc = calc_hs[a]
        x1_calc = calc_hs[b]

        if x0_calc - eps <= x <= x1_calc + eps:
            x0_web = web_hs[a]
            x1_web = web_hs[b]

            t = (x - x0_calc) / (x1_calc - x0_calc)
            t = np.clip(t, 0.0, 1.0)

            return x0_web + t * (x1_web - x0_web)

    return np.nan


x_web_like = np.array([map_calc_to_web_x(x) for x in x_calc])


# -------------------------
# 7. 실험 점 CSV 읽기
# -------------------------
exp = np.loadtxt(experiment_csv_path, delimiter=",", skiprows=1)
x_exp = exp[:, 0]
y_exp = exp[:, 1]


# -------------------------
# 8. segment index 만들기
# -------------------------
def make_segments(data, x_calc, tol=1e-10):
    """
    band.yaml에 segment_nqpoint가 있으면 그걸 우선 사용.
    없으면 distance가 같은 연속 지점을 고대칭점 경계로 보고 segment를 나눔.
    """

    if "segment_nqpoint" in data:
        segment_nqpoint = data["segment_nqpoint"]

        if sum(segment_nqpoint) == len(x_calc):
            segments = []
            start = 0

            for n in segment_nqpoint:
                end = start + n
                segments.append(np.arange(start, end))
                start = end

            return segments

    break_indices = np.where(np.isclose(np.diff(x_calc), 0.0, atol=tol))[0] + 1
    segments = np.split(np.arange(len(x_calc)), break_indices)

    return [seg for seg in segments if len(seg) > 1]


segments = make_segments(data, x_calc)


# -------------------------
# 9. overlay plot
# -------------------------
plt.figure(figsize=(6, 4))

# phonopy bands: segment별로 끊어서 그림
for seg in segments:
    valid = np.isfinite(x_web_like[seg])
    seg = seg[valid]

    if len(seg) < 2:
        continue

    for i in range(freqs.shape[1]):
        plt.plot(
            x_web_like[seg],
            freqs[seg, i],
            linewidth=1,
            color="black"
        )

# experimental points
plt.scatter(x_exp, y_exp, s=25, marker="o", label="Experiment")

# high symmetry vertical lines
for label in path:
    plt.axvline(web_hs[label], linewidth=0.6, linestyle="--", color="black")

plt.xticks(
    [web_hs[k] for k in path],
    [r"$\Gamma$", "X", "K", r"$\Gamma$", "L"]
)

plt.xlabel("Wave vector")
plt.ylabel("Frequency (THz)")
plt.xlim(web_hs["G"], web_hs["L"])
plt.tight_layout()
Path(output_png_path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_png_path, dpi=300)
plt.close()

print(f"\nSaved plot: {output_png_path}")
