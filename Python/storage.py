import pickle


def write(name, obj):
    with open(f"{name}.txt", "w") as f:
        pickle.dump(obj, f)


def read(name):
    with open(f"{name}.txt", "r") as f:
        return pickle.load(f)
