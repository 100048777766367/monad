import os
import json
import time
import subprocess
import hashlib
from cells.neuron import AtomNeuron, sigmoid_derivative

NUM_CELLS = 8
STATE_LOG = "synapses/state.json"

# ==============================================================
# HỆ THỐNG ĐA XÚC TU (1 + N SENSORY TENTACLES)
# ==============================================================
# Xúc tu tự động vươn vòi tìm tất cả các giác quan trong hệ sinh thái!
SENSORY_ORGANS = ["retina", "cochlea", "tactile", "olfactory"]

class MonadEngine:
    def __init__(self):
        # Bầy nơ-ron nhận đầu vào đa giác quan (4 chiều)
        self.swarm = [AtomNeuron(f"h_{i}", n_inputs=4, n_super_xor=4) for i in range(NUM_CELLS)]
        self.out_cell = AtomNeuron("out", n_inputs=NUM_CELLS, n_super_xor=4)

    # 1. ĐA XÚC TU VƯƠN VÒI (HÚT CÙNG LÚC MỌI GIÁC QUAN THEO PHẢ HỆ)
    def harvest_all_tentacles(self):
        owner = os.environ.get("REPO_OWNER", "")
        parent = os.environ.get("REPO_PARENT", "")
        seed = "100048777766367"
        
        harvested_signals = {}

        for organ in SENSORY_ORGANS:
            target_file = f"synapse_{organ}.json"
            candidates = [
                f"https://raw.githubusercontent.com/{owner}/{organ}/main/synapse_forward.json" if owner else None,
                f"https://raw.githubusercontent.com/{parent}/{organ}/main/synapse_forward.json" if parent else None,
                f"https://raw.githubusercontent.com/{seed}/{organ}/main/synapse_forward.json"
            ]

            for url in candidates:
                if not url: continue
                res = subprocess.run(["curl", "-s", "-f", url, "-o", target_file], capture_output=True)
                if res.returncode == 0 and os.path.exists(target_file):
                    try:
                        with open(target_file, "r") as f:
                            data = json.load(f)
                        raw_features = data.get("features", [0.5, 0.5])[:2]
                        harvested_signals[organ] = [float(x) for x in raw_features]
                        print(f"🎯 [XÚC TU {organ.upper()}] Hút thành công: {harvested_signals[organ]}")
                        break
                    except Exception: pass

        return harvested_signals

    # 2. VẬN HÀNH SIÊU XOR ĐA THỰC TẠI & HỢP NHẤT ĐA GIÁC QUAN
    def run(self):
        harvested = self.harvest_all_tentacles()
        timestamp = int(time.time())

        # Nếu không có xúc tu nào bắt được sóng, dùng tín hiệu mặc định
        if not harvested:
            print("🍃 [MÔI TRƯỜNG TĨNH] Không xúc tu nào bắt được tín hiệu. Dùng xung nội suy mặc định.")
            harvested["default_sensor"] = [0.5, 0.5]

        # ÁP DỤNG 1 + N SIÊU XOR KHÁC NHAU CHO TỪNG XÚC TU
        super_vector = []
        all_masks = {}

        for organ, signal in harvested.items():
            # Từng giác quan đi qua một ma trận Siêu XOR độc lập của nơ-ron h_0
            entangled, masks = self.swarm[0].super_xor_entangle(signal, organ)
            super_vector.extend(entangled[:2]) # Lấy 2 đặc trưng đã bẻ cong
            all_masks[organ] = masks
            print(f"🪞 [1+N SIÊU XOR {organ.upper()}] Tín hiệu thực {signal} -> Khúc xạ thành: {entangled[:2]}")

        # Chuẩn hóa vector hợp nhất thành đúng 4 chiều đầu vào cho Não
        while len(super_vector) < 4:
            super_vector.append(0.0)
        super_vector = super_vector[:4]

        print(f"⚡ [HỢP NHẤT ĐA GIÁC QUAN]: {super_vector}")

        # BẦY NƠ-RON PHÁN QUYẾT TRÊN KHÔNG GIAN SIÊU XOR
        h_outs = [cell.forward(super_vector) for cell in self.swarm]
        pred = self.out_cell.forward(h_outs)
        print(f"🔮 [PHÁN QUYẾT TỐI THƯỢNG]: {pred:.4f} (Độ tin cậy: {abs(pred - 0.5)*200:.2f}%)")

        # LAN TRUYỀN NGƯỢC (TÍNH ĐẠO HÀM ĐỂ BẮN TRẢ VỀ CHO TỪNG XÚC TU)
        target = 1.0 if super_vector[0] != super_vector[1] else 0.0
        delta_out = (target - pred) * sigmoid_derivative(pred)
        deltas_h = [delta_out * self.out_cell.weights[j] * sigmoid_derivative(h_outs[j]) for j in range(NUM_CELLS)]

        # Tự cập nhật bầy
        self.out_cell.backward(delta_out)
        for j, cell in enumerate(self.swarm):
            cell.backward(deltas_h[j])

        # Đạo hàm nén gửi ngược lại cho các giác quan
        grad_backward = [
            round(sum(deltas_h[j] * self.swarm[j].weights[0] for j in range(NUM_CELLS)), 4),
            round(sum(deltas_h[j] * self.swarm[j].weights[1] for j in range(NUM_CELLS)), 4)
        ]

        # ĐÓNG GÓI BẢN MÃ KHÁNG LƯỢNG TỬ SHAKE-256 KÈM MẶT NẠ SIÊU XOR
        feedback_packet = {
            "reply_timestamp": timestamp,
            "processed_at": time.time(),
            "gradients_for_sensory": grad_backward,
            "active_tentacles": list(harvested.keys()),
            "super_xor_masks": all_masks, # Bẫy mê cung gương cho kẻ nghe trộm
            "quantum_cid": hashlib.shake_256(f"{pred}_{timestamp}".encode()).hexdigest(64)
        }

        with open("synapse_backward.json", "w") as f:
            json.dump(feedback_packet, f, indent=4)

        with open(STATE_LOG, "w") as f:
            json.dump({"last_t": timestamp, "status": "synced"}, f, indent=4)

        print(f"✅ [HOÀN TẤT CHU KỲ] Đã bắn đạo hàm cho tất cả xúc tu: {grad_backward}")

if __name__ == "__main__":
    engine = MonadEngine()
    engine.run()
