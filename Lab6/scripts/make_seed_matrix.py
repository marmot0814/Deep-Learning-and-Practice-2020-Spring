import torch

seed1 = torch.zeros(24).long()
seed2 = torch.zeros(24, 24).long()
seed3 = torch.zeros(24, 24, 24).long()

perform1 = torch.zeros(24).float()
perform2 = torch.zeros(24, 24).float()
perform3 = torch.zeros(24, 24, 24).float()

with open('data', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        arr = line.split(':')
        seed = int(arr[1].split(", ")[0].strip())
        if seed == -1:
            seed = 0
        label = tuple([ int(x) for x in arr[2].split(", a")[0].strip()[1:-1].split(', ') ])
        acc = float(arr[3].strip())
        if len(label) == 1:
            seed1[label] = seed
            perform1[label] = acc
        elif len(label) == 2:
            seed2[label] = seed
            perform2[label] = acc
        elif len(label) == 3:
            seed3[label] = seed
            perform3[label] = acc

    torch.save(seed1, 'seed1.pth')
    torch.save(seed2, 'seed2.pth')
    torch.save(seed3, 'seed3.pth')
    torch.save(perform1, 'perform1.pth')
    torch.save(perform2, 'perform2.pth')
    torch.save(perform3, 'perform3.pth')
