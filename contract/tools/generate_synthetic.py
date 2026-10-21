"""Generate a synthetic side-on training video's pipeline data: two instep shots.

Rep r001: right foot, good technique.
Rep r002: left foot (weak foot), planted too far from the ball and leaning back.

Writes into examples/:
  pose.example.json, ball.example.json   (with pixel noise, like a real detector)
  reps.example.json, labels.example.csv  (ground truth: exact phase frames + kicking foot)
  metrics.example.json                   (ground truth, measured on the noise-free skeleton)

Because the true answers are known, every workstream can test against these files
before real footage exists. Run:  python tools/generate_synthetic.py
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

FPS = 120
W, H_IMG = 1920, 1080
GROUND = 940.0          # ankle height (y) when standing
THIGH = SHIN = 165.0    # px
TORSO = 200.0
NECK_TO_NOSE = 70.0
BALL_R = 39.0           # size-5 ball (22 cm) at this player's scale (~354 px per metre)
BALL_Y = GROUND + 15 - BALL_R   # ball resting on the ground (soles sit ~15 px below the ankle joint)
HEIGHT_CM = 175
OUT = Path(__file__).resolve().parent.parent / "examples"
rng = np.random.default_rng(7)

KEYPOINTS = [
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def ang(deg):
    """Unit vector at `deg` degrees from straight down, positive = forward (+x)."""
    r = math.radians(deg)
    return np.array([math.sin(r), math.cos(r)])


def ik_knee(hip, ankle):
    """Two-link IK; the knee bends forward (+x), as a human knee does."""
    d = ankle - hip
    dist = min(np.linalg.norm(d), THIGH + SHIN - 1e-3)
    base = math.degrees(math.atan2(d[0], d[1]))          # angle of hip->ankle from down
    bend = math.degrees(math.acos((THIGH**2 + dist**2 - SHIN**2) / (2 * THIGH * dist)))
    return hip + THIGH * ang(base + bend)


def lerp(a, b, s):
    return a + (b - a) * s


def smooth(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


def make_rep(kick, ball_x, plant_gap, lean_back, ball_speed_px, hip_back, ankle_plantar):
    """Return (frames, ball_track, truth) for one rep. Frame indices are rep-local."""
    sup = "left" if kick == "right" else "right"
    P = 96                       # plant frame (rep-local)
    C = P + 7                    # contact ~58 ms after plant
    E = C + 40                   # end of follow-through
    N = E + 25
    plant_x = ball_x - plant_gap
    step = 28                    # frames per step during approach

    # Footfalls: (frame, x). Support foot lands last at plant_x; feet alternate.
    sup_falls, kick_falls = [(P, plant_x)], []
    x, f, who = plant_x, P, "kick"
    while f > -step:
        f -= step
        x -= 95
        (kick_falls if who == "kick" else sup_falls).append((f, x))
        who = "sup" if who == "kick" else "kick"
    sup_falls.sort(); kick_falls.sort()
    K0 = kick_falls[-1][0]      # kicking foot's last footfall before the kick

    def foot_pos(falls, t, stance=16):
        prev = max([fl for fl in falls if fl[0] <= t], default=falls[0])
        nxt = min([fl for fl in falls if fl[0] > t], default=None)
        if nxt is None or t - prev[0] < stance:
            return np.array([prev[1], GROUND])
        s = (t - prev[0] - stance) / (nxt[0] - prev[0] - stance)
        return np.array([lerp(prev[1], nxt[1], smooth(s)), GROUND - 45 * math.sin(math.pi * s)])

    from scipy.interpolate import CubicSpline

    def hip_at(t):
        if t <= P:
            x = plant_x - hip_back - (P - t) * (95 / step)
        else:
            x = plant_x - hip_back + 30 * smooth((t - P) / (E - P))
        bob = 6 * math.sin(2 * math.pi * t / step) if t < P else 0
        y = GROUND - (THIGH + SHIN) * 0.93 + bob + (12 if P <= t <= C + 10 else 0)
        return np.array([x, y])

    # Kicking ankle follows a smooth path: lift -> backswing -> bottom of swing -> contact -> follow-through.
    # A spline (not keyframe easing) keeps the foot fastest at contact, as in a real strike.
    t_lift = K0 + 16
    keys = [
        (t_lift, foot_pos(kick_falls, t_lift)),
        (P, hip_at(P) + [-150, 190]),
        (P + 5, np.array([ball_x - 120, GROUND - 12])),
        (C, np.array([ball_x - 32, BALL_Y - 6])),
        (E, hip_at(E) + [210, 40]),
    ]
    kick_path = CubicSpline([k[0] for k in keys], np.array([k[1] for k in keys]), axis=0)

    frames, balls = [], []
    for t in range(N):
        hip = hip_at(t)

        # Support leg
        s_ank = foot_pos(sup_falls, t) if t < P else np.array([plant_x, GROUND])
        s_knee = ik_knee(hip, s_ank)
        s_foot = 0.0

        # Kicking leg
        if t <= t_lift:
            k_ank = foot_pos(kick_falls, t)
            k_foot = 0.0
        else:
            k_ank = kick_path(min(t, E))
            k_foot = ankle_plantar if t >= P else 15.0   # toes pointed (ankle locked) through the strike
        k_knee = ik_knee(hip, k_ank)

        def foot(ankle, knee, plantar):
            shin_deg = math.degrees(math.atan2(*(ankle - knee)))
            heel = ankle + 22 * ang(shin_deg - 90 + plantar)  # back of foot
            toe = ankle + 48 * ang(shin_deg + 90 - plantar)
            return heel, toe

        s_heel, s_toe = foot(s_ank, s_knee, s_foot)
        k_heel, k_toe = foot(k_ank, k_knee, k_foot)

        lean = 6 - (6 + lean_back) * smooth((t - (P - 20)) / 25) if t > P - 20 else 6
        shoulder = hip + TORSO * np.array([math.sin(math.radians(lean)), -math.cos(math.radians(lean))])
        nose = shoulder + NECK_TO_NOSE * np.array([math.sin(math.radians(lean + 10)), -1])
        swing = 35 * math.sin(2 * math.pi * t / step) if t < P else -40   # opposite arm counter-balances
        elb_k = shoulder + 120 * ang(swing); wri_k = elb_k + 110 * ang(swing + 30)
        elb_s = shoulder + 120 * ang(-swing); wri_s = elb_s + 110 * ang(-swing + 30)

        kp = {
            "nose": nose,
            f"{kick}_shoulder": shoulder + [4, 0], f"{sup}_shoulder": shoulder - [4, 0],
            f"{kick}_elbow": elb_k, f"{sup}_elbow": elb_s,
            f"{kick}_wrist": wri_k, f"{sup}_wrist": wri_s,
            f"{kick}_hip": hip + [3, 0], f"{sup}_hip": hip - [3, 0],
            f"{kick}_knee": k_knee, f"{sup}_knee": s_knee,
            f"{kick}_ankle": k_ank, f"{sup}_ankle": s_ank,
            f"{kick}_heel": k_heel, f"{sup}_heel": s_heel,
            f"{kick}_foot_index": k_toe, f"{sup}_foot_index": s_toe,
        }
        frames.append(kp)

        # Ball: at rest until contact, then flies forward and up with gravity.
        if t <= C:
            b = np.array([ball_x, BALL_Y])
        else:
            dt = t - C
            b = np.array([ball_x + ball_speed_px * 0.97 * dt, BALL_Y - ball_speed_px * 0.25 * dt + 0.15 * dt * dt])
        balls.append(b if 0 <= b[0] < W and 0 <= b[1] < H_IMG else None)

    truth = {"foot": kick, "start": 0, "plant": P, "contact": C, "end": E, "length": N}
    return frames, balls, truth


def body_height_px(frames):
    """Contract definition: nose y to the lower ankle y, median over the first 30 frames."""
    hs = [max(f["left_ankle"][1], f["right_ankle"][1]) - f["nose"][1] for f in frames[:30]]
    return float(np.median(hs))


def angle_at(a, b, c):
    v1, v2 = a - b, c - b
    return math.degrees(math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))


def true_metrics(frames, balls, tr, offset, ball_speed_px):
    k, s = tr["foot"], ("left" if tr["foot"] == "right" else "right")
    P, C, E = tr["plant"], tr["contact"], tr["end"]
    Hpx = body_height_px(frames)
    fp, fc, fe = frames[P], frames[C], frames[E]
    ball = balls[C]
    hip_mid = (fc["left_hip"] + fc["right_hip"]) / 2
    sh_mid = (fc["left_shoulder"] + fc["right_shoulder"]) / 2
    trunk = math.degrees(math.atan2(sh_mid[0] - hip_mid[0], hip_mid[1] - sh_mid[1]))  # + = forward
    px_per_m = Hpx / (HEIGHT_CM / 100 * 0.93)   # nose-to-ankle is ~93% of standing height
    speed_ms = ball_speed_px * math.hypot(0.97, 0.25) * FPS / px_per_m
    return [
        ("plant_foot_distance", abs(fp[f"{s}_ankle"][0] - balls[P][0]) / Hpx, "body_heights", P),
        ("plant_foot_angle", None, "degrees", P),
        ("knee_over_ball", (fc[f"{k}_knee"][0] - ball[0]) / Hpx, "body_heights", C),
        ("trunk_lean", trunk, "degrees", C),
        ("hip_shoulder_separation", None, "degrees", P),
        ("ankle_lock", angle_at(fc[f"{k}_knee"], fc[f"{k}_ankle"], fc[f"{k}_foot_index"]), "degrees", C),
        ("follow_through", (fe[f"{k}_ankle"][0] - fc[f"{k}_ankle"][0]) / Hpx, "body_heights", E),
        ("ball_speed", speed_ms, "m_per_s", C + 3),
    ], offset


def main():
    import yaml
    targets = yaml.safe_load((OUT / "targets.example.yaml").read_text())["shot_instep"]

    reps_cfg = [
        dict(rep_id="r001", kick="right", ball_x=1150, plant_gap=75, lean_back=0, ball_speed_px=60, hip_back=-10, ankle_plantar=65),
        dict(rep_id="r002", kick="left", ball_x=1150, plant_gap=135, lean_back=16, ball_speed_px=40, hip_back=40, ankle_plantar=40),
    ]
    GAP = 60   # frames with nobody detected between reps (player walking back out of frame)

    pose_frames, ball_frames, reps, labels, metrics_out = [], [], [], [], []
    for i, cfg in enumerate(reps_cfg):
        rep_id = cfg.pop("rep_id")
        frames, balls, tr = make_rep(**cfg)
        offset = len(pose_frames)
        mets, _ = true_metrics(frames, balls, tr, offset, cfg["ball_speed_px"])

        for t, (kp, b) in enumerate(zip(frames, balls)):
            noisy = {}
            for name in KEYPOINTS:
                conf = float(np.clip(rng.normal(0.93, 0.04), 0.5, 0.99))
                if name.endswith(("heel", "foot_index")):
                    conf = float(np.clip(conf - 0.12, 0.3, 0.99))    # feet are the least reliable joints
                x, y = kp[name] + rng.normal(0, 1.8, 2)
                noisy[name] = [round(float(x), 1), round(float(y), 1), round(conf, 2)]
            pose_frames.append({"f": offset + t, "keypoints": noisy})
            near_foot = abs(t - tr["contact"]) <= 1
            if b is None or (near_foot and i == 1):   # rep 2: ball hidden by the foot at contact, like real footage
                ball_frames.append({"f": offset + t, "ball": None})
            else:
                bx, by = b + rng.normal(0, 1.2, 2)
                ball_frames.append({"f": offset + t, "ball": [round(float(bx), 1), round(float(by), 1), BALL_R, round(float(rng.uniform(0.8, 0.97)), 2)]})

        for g in range(GAP if i < len(reps_cfg) - 1 else 0):
            f = len(pose_frames)
            pose_frames.append({"f": f, "keypoints": {}})
            ball_frames.append({"f": f, "ball": None})

        fr = {k: offset + tr[k] for k in ("start", "plant", "contact", "end")}
        reps.append({"rep_id": rep_id, "move": "shot_instep", "foot": tr["foot"], "frames": fr, "confidence": 1.0})
        labels.append({"video": "synthetic_shots.mp4", "rep_id": rep_id, "move": "shot_instep",
                       "foot": tr["foot"], "plant_frame": fr["plant"], "contact_frame": fr["contact"],
                       "coach_score_1to5": 4 if i == 0 else 2, "notes": "synthetic ground truth"})

        mlist = []
        for name, value, unit, f_local in mets:
            tgt = targets[name]
            if value is None:
                status, val = "not_measured", None
            else:
                val = round(value, 3 if unit == "body_heights" else 1)
                status = "in_range" if tgt["range"][0] <= val <= tgt["range"][1] else "out_of_range"
            mlist.append({"name": name, "value": val, "unit": unit, "target": tgt["range"],
                          "status": status, "frame": offset + f_local})
        metrics_out.append({"rep_id": rep_id, "move": "shot_instep", "foot": tr["foot"], "metrics": mlist})

    meta = {"video": "synthetic_shots.mp4", "fps": FPS, "width": W, "height": H_IMG}
    dump = lambda name, obj: (OUT / name).write_text(json.dumps(obj, separators=(",", ":")) if name.startswith(("pose", "ball")) else json.dumps(obj, indent=2))
    dump("pose.example.json", {**meta, "schema_version": "1.0", "player_track_id": 1, "frames": pose_frames})
    dump("ball.example.json", {**meta, "schema_version": "1.0", "frames": ball_frames})
    dump("reps.example.json", {"schema_version": "1.0", "video": meta["video"], "player_id": "p1", "reps": reps})
    dump("metrics.example.json", {"schema_version": "1.0", "video": meta["video"], "player_id": "p1", "reps": metrics_out})
    with open(OUT / "labels.example.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(labels[0]))
        wr.writeheader(); wr.writerows(labels)

    print(f"{len(pose_frames)} frames written to {OUT}")
    for r, m in zip(reps, metrics_out):
        print(r["rep_id"], r["foot"], r["frames"])
        for x in m["metrics"]:
            print(f"   {x['name']:<24} {x['value']!s:>8} {x['unit']:<13} target {x['target']}  {x['status']}")


if __name__ == "__main__":
    main()
