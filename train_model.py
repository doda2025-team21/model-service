import time
from pathlib import Path

def train():
    model = f"Trained model placeholder - {time.ctime()}"
    Path("models").mkdir(exist_ok=True)
    filename = f"models/model-{int(time.time())}.bin"
    with open(filename, "w") as f:
        f.write(model)
    print(filename)

if __name__ == "__main__":
    train()
