import math
import random
import json
import os

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(min(x, 500), -500)))

def sigmoid_derivative(y):
    return y * (1.0 - y)

class AtomNeuron:
    def __init__(self, neuron_id, n_inputs=2, storage_dir="synapses"):
        self.id = neuron_id
        self.n_in = n_inputs
        self.storage_path = os.path.join(storage_dir, f"{self.id}.json")
        self.os_dir = storage_dir
        
        self.weights = []
        self.bias = 0.0
        self.last_inputs = []
        self.last_out = 0.0
        
        self.load_self()

    # TỰ CHỮA LÀNH (SELF-HEALING): Mất file nhớ tự sinh lại trong 0.001s
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

    # PHÉP TÍNH THỨ 5 + KHÁNG ĐỘC TỐ (Miễn nhiễm NaN / Vô cực)
    def forward(self, inputs):
        # Lọc sạch độc tố đầu vào: Ép vào khoảng an toàn [-10.0, 10.0]
        self.last_inputs = [max(-10.0, min(10.0, float(x))) for x in inputs[:self.n_in]]
        tong = sum(self.last_inputs[i] * self.weights[i] for i in range(self.n_in)) + self.bias
        self.last_out = sigmoid(tong)
        return self.last_out

    # LAN TRUYỀN NGƯỢC (TỰ HẤP THỤ NĂNG LƯỢNG ĐỐI KHÁNG)
    def backward(self, delta, lr=0.1):
        # Giới hạn delta chống nổ gradient
        safe_delta = max(-5.0, min(5.0, delta))
        for i in range(self.n_in):
            self.weights[i] += lr * safe_delta * self.last_inputs[i]
        self.bias += lr * safe_delta
        self.save_self()

    def save_self(self):
        os.makedirs(self.os_dir, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({"w": [round(w, 4) for w in self.weights], "b": round(self.bias, 4)}, f, indent=4)
