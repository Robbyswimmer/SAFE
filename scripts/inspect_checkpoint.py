import torch
import sys

def inspect_checkpoint(path):
    print("Loading " + path + "...")
    ckpt = torch.load(path, map_location="cpu")
    if "state_dict" in ckpt:
        keys = list(ckpt["state_dict"].keys())
    else:
        keys = list(ckpt.keys())
        
    print("Total keys: " + str(len(keys)))
    print("First 20 keys:")
    for k in keys[:20]:
        print("  " + k)
        
    print("\nSearching for 'audio_projector':")
    proj_keys = [k for k in keys if "audio_projector" in k]
    for k in proj_keys[:10]:
        print("  " + k)

if __name__ == "__main__":
    inspect_checkpoint(sys.argv[1])
