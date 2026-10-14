import os
import json
import time
import subprocess
import hashlib
from cells.neuron import AtomNeuron, sigmoid_derivative

NUM_CELLS = 8
STATE_LOG = "synapses/state.json"
SIGNAL_FILE = "synapse_signal.json"

class MonadEngine:
    def __init__(self):
        # Bầy nơ-ron nguyên tử phân tán (không có tệp trùm)
        self.swarm = [AtomNeuron(f"h_{i}", n_inputs=2) for i in range(NUM_CELLS)]
        self.out_cell = AtomNeuron("out", n_inputs=NUM_CELLS)

    # 1. XÚC TU VƯƠN VÒI (CASCADE TRY-FALLBACK): Tự dò tìm môi trường
    def sniff_environment(self):
        owner = os.environ.get("REPO_OWNER", "")
        parent = os.environ.get("REPO_PARENT", "")
        seed = "100048777766367"
        
        candidates = [
            f"https://raw.githubusercontent.com/{owner}/retina/main/synapse_forward.json" if owner else None,
            f"https://raw.githubusercontent.com/{parent}/retina/main/synapse_forward.json" if parent else None,
            f"https://raw.githubusercontent.com/{seed}/retina/main/synapse_forward.json"
        ]

        for url in candidates:
            if not url: continue
            res = subprocess.run(["curl", "-s", "-f", url, "-o", SIGNAL_FILE], capture_output=True)
            if res.returncode == 0 and os.path.exists(SIGNAL_FILE):
                print(f"🎯 [HÚT XUNG MÔI TRƯỜNG] Bắt được tín hiệu từ: {url}")
                return True
        return False

    # 2. VẬN HÀNH BẤT ĐỒNG BỘ & SIÊU XOR
    def run(self):
        has_signal = self.sniff_environment()
        features = [0.5, 0.5]
        timestamp = int(time.time())

        if has_signal:
            try:
                with open(SIGNAL_FILE, "r") as f:
                    data = json.load(f)
                features = [float(x) for x in data.get("features", [0.5, 0.5])[:2]]
                timestamp = data.get("timestamp", timestamp)
            except Exception: pass

        # Kiểm tra chống chạy lặp (Idempotency)
        last_time = 0
        if os.path.exists(STATE_LOG):
            try:
                with open(STATE_LOG, "r") as f: last_time = json.load(f).get("last_t", 0)
            except Exception: pass

        if timestamp > 0 and timestamp <= last_time and has_signal:
            print("💤 [BỎ QUA] Tín hiệu đã được tiêu hóa trước đó. Đi ngủ tiếp.")
            return

        print(f"⚡ [VẬN HÀNH MONAD] Xử lý tín hiệu: {features}")

        # FORWARD: Bầy nơ-ron tính toán
        h_outs = [cell.forward(features) for cell in self.swarm]
        pred = self.out_cell.forward(h_outs)
        print(f"🔮 [PHÁN QUYẾT BẦY]: {pred:.4f} (Độ tin cậy: {abs(pred - 0.5)*200:.2f}%)")

        # BACKWARD & BẮN ĐẠO HÀM (SPLIT LEARNING)
        target = 1.0 if features[0] != features[1] else 0.0 # Logic XOR tự nhiên
        delta_out = (target - pred) * sigmoid_derivative(pred)
        deltas_h = [delta_out * self.out_cell.weights[j] * sigmoid_derivative(h_outs[j]) for j in range(NUM_CELLS)]

        # Tự cập nhật bầy
        self.out_cell.backward(delta_out)
        for j, cell in enumerate(self.swarm):
            cell.backward(deltas_h[j])

        # Phép tính đạo hàm gửi ngược lại cho giác quan
        grad_backward = [
            round(sum(deltas_h[j] * self.swarm[j].weights[0] for j in range(NUM_CELLS)), 4),
            round(sum(deltas_h[j] * self.swarm[j].weights[1] for j in range(NUM_CELLS)), 4)
        ]

        # ĐÓNG GÓI BẢN MÃ KHÁNG LƯỢNG TỬ (SHAKE-256)
        feedback_packet = {
            "reply_timestamp": timestamp,
            "processed_at": time.time(),
            "gradients_for_sensory": grad_backward,
            "monad_cid": hashlib.shake_256(f"{pred}_{timestamp}".encode()).hexdigest(32)
        }

        with open("synapse_backward.json", "w") as f:
            json.dump(feedback_packet, f, indent=4)

        with open(STATE_LOG, "w") as f:
            json.dump({"last_t": timestamp, "status": "synced"}, f, indent=4)

        print(f"✅ [HOÀN TẤT CHU KỲ] Đã để lại vết tích tiến hóa: {grad_backward}")

if __name__ == "__main__":
    engine = MonadEngine()
    engine.run()
