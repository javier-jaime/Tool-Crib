# Import necessary libraries
from rectpack import newPacker
import rectpack.packer as packer
import matplotlib.pyplot as plt

# Function Solver
def solver(list, trucks):

    # Build the Packer
    pack = newPacker(mode = packer.PackingMode.Offline, bin_algo = packer.PackingBin.Global,
                     rotation=True)

    # Add the rectangles to packing queue
    for l in list:
        pack.add_rect(*l)

    # Add the bins where the rectangles will be placed
    for t in trucks:
        pack.add_bin(*t)

    # Start packing
    pack.pack()

    # Full rectangle list from furniture
    all_rects = pack.rect_list()

    # Furniture with dimensions
    all_furniture = [sorted([p[3], p[4]]) for p in all_rects]

    # Count pieces of furniture
    furniture = all_furniture.count(list)
    print(furniture)

    # Number of rectangles packed into first bin
    nrect = len(pack[0])

    return all_rects, all_furniture, nrect
    
def plot_solution(all_rects):
    # Plot
    plt.figure(figsize=(10,20))
    # Loop all rect
    for rect in all_rects:
        b, x, y, w, h, rid = rect
        x1, x2, x3, x4, x5 = x, x+w, x+w, x, x
        y1, y2, y3, y4, y5 = y, y, y+h, y+h,y

        plt.plot([x1,x2,x3,x4,x5],[y1,y2,y3,y4,y5])

    plt.show()