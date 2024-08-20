import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

def visualization_with_zoom(x, zoom, colorbar, name):
    a = 48
    b = a + 30
    c = 73
    d = c + 30
    
    vmin = np.min(x)
    vmax = np.max(x)

    fig, ax = plt.subplots()
    plt.imshow(x, cmap='gray', vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar()
    plt.axis('off')
    
    if zoom:
        rect = patches.Rectangle((c, a), d-c, b-a, linewidth=1.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # sub_axes = plt.axes([.55, .62, .25, .25]) 
        sub_axes = plt.axes([.2, .12, .25, .25]) 
        for axis in ['top','bottom','left','right']:
            sub_axes.spines[axis].set_linewidth(1.5)
        sub_axes.imshow(x[a:b, c:d], cmap='gray', vmin=vmin, vmax=vmax) 
        sub_axes.spines['bottom'].set_color('red')
        sub_axes.spines['top'].set_color('red')
        sub_axes.spines['left'].set_color('red')
        sub_axes.spines['right'].set_color('red')
        sub_axes.set_xticks([])
        sub_axes.set_yticks([])
        
    if len(name) > 0:
        plt.savefig(name, bbox_inches='tight', dpi=1200)
