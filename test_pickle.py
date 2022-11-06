import pickle

with open("fsp_losses.pickle", "rb") as f:
    bin_arch, fsp = pickle.load(f)[0]
    print("----", bin_arch)
    print("++++", fsp)
