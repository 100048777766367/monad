import math
import random
import json
import os
import hashlib

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(min(x, 500), -500)))

def sigmoid_derivative(y):
    return y * (1.0 - y)

class AtomNeuron:
    def __init__(self, neuron_id, n_inputs=4, n_super_xor=4, storage_dir="synapses"):
        self.id = neuron_id
        self.n_in = n_inputs
        self.n_xor = n_super_xor # Số chiều thực tại Siêu XOR (1 + N)
        self.storage_path = os.path.join(storage_dir, f"{self.id}.json")
        self.os_dir = storage_dir
        
        self.weights = []
        self.bias = 0.0
        self.last_inputs = []
        self.last_out = 0.0
        
        self.load_self()

    def load_self(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                if len(data.get("w", [])) == self.n_in:
                    self.weights = data["w"]
                    self.bias = data.get("b", 0.0)
                    return
            except Exception: pass

        self.weights = [random.uniform(-1.0, 1.0) for _ in range(self.n_in)]
        self.bias = random.uniform(-0.5, 0.5)
        self.save_self()

    # ==============================================================
    # MA TRẬN 1 + N SIÊU XOR NGUYÊN TỬ: BẺ CONG ĐA THỰC TẠI
    # ==============================================================
    def super_xor_entangle(self, stimulus_vector, organ_id):
        """
        Mỗi giác quan (Mắt, Tai, Da) sẽ có một bộ mặt nạ Siêu XOR KHÁC NHAU hoàn toàn!
        """
        masks = []
        for d in range(self.n_xor):
            # Tạo mặt nạ kháng lượng tử từ SHAKE-256 kết hợp mã giác quan
            seed = f"{self.id}_{organ_id}_dim_{d}".encode()
            mask_val = int(hashlib.shake_256(seed).hexdigest(4), 16) / 65535.0
            masks.append(round(mask_val, 4))

        # Phép Siêu XOR thực số: Chồng chập liên tục qua N thực tại
        entangled = list(stimulus_vector)
        for i in range(len(entangled)):
            for m in masks:
                entangled[i] = round((entangled[i] + m) % 1.0, 4)
                
        return entangled, masks

    # LAN TRUYỀN TIẾN: Nhận tín hiệu đã qua Siêu XOR và tính toán
    def forward(self, inputs):
        # Kháng độc tố: Giới hạn an toàn tuyệt đối
        self.last_inputs = [max(-10.0, min(10.0, float(x))) for x in inputs[:self.n_in]]
        # Bù thêm 0 nếu đầu vào ngắn hơn số nơ-ron
        while len(self.last_inputs) < self.n_in:
            self.last_inputs.append(0.0)

        tong = sum(self.last_inputs[i] * self.weights[i] for i in range(self.n_in)) + self.bias
        self.last_out = sigmoid(tong)
        return self.last_out

    # LAN TRUYỀN NGƯỢC: Tự tiến hóa trọng số
    def backward(self, delta, lr=0.1):
        safe_delta = max(-5.0, min(5.0, delta))
        for i in range(self.n_in):
            self.weights[i] += lr * safe_delta * self.last_inputs[i]
        self.bias += lr * safe_delta
        self.save_self()

    def save_self(self):
        os.makedirs(self.os_dir, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({"w": [round(w, 4) for w in self.weights], "b": round(self.bias, 4)}, f, indent=4)
