import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_4DLine(cloud, versor, pa, title = ""):
    fig = plt.figure(dpi = 200)
    ax = plt.axes(projection="3d")
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('t')
    ax.set_zlabel('Y')
    ax.set_xlim((-140, 140))
    ax.set_ylim((0, 512))
    ax.set_zlim((-140, 140))
    ax.view_init(None, 75)
    i = 0
    colors = ['b', 'r', 'green', 'orange', 'm', 'k', 'y', 'c', 'tan', 'salmon']
    reta_centro = [[0, 0, 0], [0, 512, 0]]
    ax.plot3D(*reta_centro, linestyle = 'dotted', lw = 2)
    data = np.copy(cloud[:, :3])
    linepts = versor * np.mgrid[-800:800:2j][:, np.newaxis]
    linepts += pa        
    x0, y0, z0, x1, y1, z1 = linepts[0][0], linepts[0][1], linepts[0][2], linepts[1][0], linepts[1][1], linepts[1][2]
    ax.plot3D(np.array([x0, x1]),np.array([z0, z1]),np.array([y0, y1]), label = str(i), color = colors[i])        
    ax.scatter3D(data[:, 0], data[:, 2], data[:, 1], marker = '.', s = 3, alpha = 1, c = colors[i])
    i += 1
    #plt.legend()
    plt.show()

def compare_clustering(cluster_before, cluster_after, number = ""):
    fig = plt.figure(figsize=plt.figaspect(0.4), dpi = 200)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    #ax1 = plt.axes(projection="3d")
    ax1.view_init(None, 75)
    ax2.view_init(None, 75)
    ax1.set_zlim((-140, 140))
    ax1.set_xlim((-140, 140))
    ax1.set_ylim((0, 512))
    ax2.set_zlim((-140, 140))
    ax2.set_xlim((-140, 140))
    ax2.set_ylim((0, 512))
    ax1.set_xlabel('X')
    ax1.set_ylabel('t')
    ax1.set_zlabel('Y')
    ax1.set_title("Before clustering: " + number)
    ax2.set_title("After clustering: " + number)
    ax2.set_xlabel('X')
    ax2.set_ylabel('t')
    ax2.set_zlabel('Y')
    # ax1.scatter3D(raw_data[:, 0], raw_data[:, 2], raw_data[:, 1], marker = '.', s = 3, alpha = 1)    
    i = 0
    colors = ['b', 'r', 'green', 'orange', 'm', 'k', 'y', 'c', 'tan', 'salmon']
    #reta_centro = [[0, 0, 0], [0, 512, 0]]
    ax1.quiver(0, 512, 0, -0, -512, -0, linestyle = 'solid', arrow_length_ratio = 0.05, lw = 1, color = 'red')
    ax2.quiver(0, 512, 0, -0, -512, -0, linestyle = 'solid', arrow_length_ratio = 0.05, lw = 1, color = 'red')
    for track in cluster_before:
        data = np.copy(track.get_data()[:, :3])
        direction = np.copy(track.get_versor())
        pb = track.get_pb()
        linepts = direction * np.mgrid[-800:800:2j][:, np.newaxis]
        linepts += pb
        x0, y0, z0, x1, y1, z1 = linepts[0][0], linepts[0][1], linepts[0][2], linepts[1][0], linepts[1][1], linepts[1][2]
        if track.isprimary:
            l = "PB"
        else:
            l = str(len(data))
        ax1.plot3D(np.array([x0, x1]),np.array([z0, z1]),np.array([y0, y1]), label = l, color = colors[i])        
        ax1.scatter3D(data[:, 0], data[:, 2], data[:, 1], marker = '.', s = 3, alpha = 1, c = colors[i])
        if track.vinchamber:
            vertex = track.get_vertex()
            ax1.scatter3D(vertex[0], vertex[2], vertex[1], marker = '^', s = 25, alpha = 0.85, c = colors[i])
        i += 1
    i = 0
    for track in cluster_after:
        data = np.copy(track.get_data()[:, :3])
        direction = np.copy(track.get_versor())
        pb = track.get_pb()
        linepts = direction * np.mgrid[-800:800:2j][:, np.newaxis]
        linepts += pb
        x0, y0, z0, x1, y1, z1 = linepts[0][0], linepts[0][1], linepts[0][2], linepts[1][0], linepts[1][1], linepts[1][2]
        if track.isprimary:
            l = "PB"
        else:
            l = str(len(data))
        ax2.plot3D(np.array([x0, x1]),np.array([z0, z1]),np.array([y0, y1]), label = l, color = colors[i])        
        ax2.scatter3D(data[:, 0], data[:, 2], data[:, 1], marker = '.', s = 3, alpha = 1, c = colors[i])
        if track.vinchamber:
            vertex = track.get_vertex()
            ax2.scatter3D(vertex[0], vertex[2], vertex[1], marker = '^', s = 25, alpha = 0.85, c = colors[i])
        i += 1
    ax2.legend(bbox_to_anchor=(1.5,1))
    ax1.legend(bbox_to_anchor=(0.1,1))
    plt.legend()
    plt.show()

def compare_clustering_ATTPC(cluster_before, cluster_after, number = ""):
    fig = plt.figure(figsize=plt.figaspect(0.4), dpi = 200)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    #ax1 = plt.axes(projection="3d")
    ax1.view_init(None, 75)
    ax2.view_init(None, 75)
    ax1.set_zlim((-250, 250))
    ax1.set_xlim((-250, 250))
    ax1.set_ylim((-200, 1200))
    ax2.set_zlim((-250, 250))
    ax2.set_xlim((-250, 250))
    ax2.set_ylim((-200, 1200))
    ax1.set_xlabel('X')
    ax1.set_ylabel('t')
    ax1.set_zlabel('Y')
    ax1.set_title("Before clustering: " + number)
    ax2.set_title("After clustering: " + number)
    ax2.set_xlabel('X')
    ax2.set_ylabel('t')
    ax2.set_zlabel('Y')
    # ax1.scatter3D(raw_data[:, 0], raw_data[:, 2], raw_data[:, 1], marker = '.', s = 3, alpha = 1)    
    i = 0
    colors = ['b', 'r', 'green', 'orange', 'm', 'k', 'y', 'c', 'tan', 'salmon']
    #reta_centro = [[0, 0, 0], [0, 512, 0]]
    ax1.quiver(0, 512, 0, -0, -512, -0, linestyle = 'solid', arrow_length_ratio = 0.05, lw = 1, color = 'red')
    ax2.quiver(0, 512, 0, -0, -512, -0, linestyle = 'solid', arrow_length_ratio = 0.05, lw = 1, color = 'red')
    for track in cluster_before:
        data = np.copy(track.get_data()[:, :3])
        direction = np.copy(track.get_versor())
        pb = track.get_pb()
        linepts = direction * np.mgrid[-800:800:2j][:, np.newaxis]
        linepts += pb
        x0, y0, z0, x1, y1, z1 = linepts[0][0], linepts[0][1], linepts[0][2], linepts[1][0], linepts[1][1], linepts[1][2]
        if track.isprimary:
            l = "PB"
        else:
            l = str(len(data))
        ax1.plot3D(np.array([x0, x1]),np.array([z0, z1]),np.array([y0, y1]), label = l, color = colors[i])        
        ax1.scatter3D(data[:, 0], data[:, 2], data[:, 1], marker = '.', s = 3, alpha = 1, c = colors[i])
        if track.vinchamber:
            vertex = track.get_vertex()
            ax1.scatter3D(vertex[0], vertex[2], vertex[1], marker = '^', s = 25, alpha = 0.85, c = colors[i])
        i += 1
    i = 0
    for track in cluster_after:
        data = np.copy(track.get_data()[:, :3])
        direction = np.copy(track.get_versor())
        pb = track.get_pb()
        linepts = direction * np.mgrid[-800:800:2j][:, np.newaxis]
        linepts += pb
        x0, y0, z0, x1, y1, z1 = linepts[0][0], linepts[0][1], linepts[0][2], linepts[1][0], linepts[1][1], linepts[1][2]
        if track.isprimary:
            l = "PB"
        else:
            l = str(len(data))
        ax2.plot3D(np.array([x0, x1]),np.array([z0, z1]),np.array([y0, y1]), label = l, color = colors[i])        
        ax2.scatter3D(data[:, 0], data[:, 2], data[:, 1], marker = '.', s = 3, alpha = 1, c = colors[i])
        if track.vinchamber:
            vertex = track.get_vertex()
            ax2.scatter3D(vertex[0], vertex[2], vertex[1], marker = '^', s = 25, alpha = 0.85, c = colors[i])
        i += 1
    ax2.legend(bbox_to_anchor=(1.5,1))
    ax1.legend(bbox_to_anchor=(0.1,1))
    plt.legend()
    plt.show()

def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    z = np.linspace(0, height_z, 150)
    theta = np.linspace(0, 2*np.pi, 150)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid

def Plot_Cluster_raw_ATTPC(cluster, number = ""):
    fig = plt.figure(figsize = plt.figaspect(1.), dpi = 200)
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    #ax1 = plt.axes(projection="3d")
    ax1.view_init(None, 75)
    ax1.set_zlim((-250, 250))
    ax1.set_xlim((-250, 250))
    ax1.set_ylim((-200, 1200))
    ax1.set_xlabel('X')
    ax1.set_ylabel('t')
    ax1.set_zlabel('Y')
    ax1.axis('off')
    # ax1.set_title(f"Event number {number}", y = 1.)
    ax1.scatter3D(cluster[:, 0], cluster[:, 2], cluster[:, 1], marker = '.', s = 3, alpha = 1)    
    #reta_centro = [[0, 0, 0], [0, 512, 0]]
    ax1.quiver(0, 512, 0, -0, -512, -0, linestyle = 'solid', arrow_length_ratio = 0.05, lw = 1, color = 'red')
    Xc, Yc, Zc = data_for_cylinder_along_z(0., 0., 250., 1000)
    ax1.plot_surface(Xc, Zc, Yc, alpha = 0.15, color = "gray")
    # ax1.legend(bbox_to_anchor =(0.1,1))
    # plt.legend()
    fig.tight_layout(rect = (0., -0.2, 1., 1.1))
    plt.show()

def Plot_Cluster_raw(cluster, number = ""):
    fig = plt.figure(figsize = plt.figaspect(1.), dpi = 200)
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    #ax1 = plt.axes(projection="3d")
    ax1.view_init(None, 75)
    ax1.set_zlim((-140, 140))
    ax1.set_xlim((-140, 140))
    ax1.set_ylim((0, 512))
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('t')
    # ax1.set_zlabel('Y')
    ax1.axis('off')
    # ax1.set_title(f"Event number {number}", y = 1.)
    ax1.scatter3D(cluster[:, 0], cluster[:, 2], cluster[:, 1], marker = '.', s = 3, alpha = 1)    
    #reta_centro = [[0, 0, 0], [0, 512, 0]]
    ax1.quiver(0, 512, 0, -0, -512, -0, linestyle = 'solid', arrow_length_ratio = 0.05, lw = 1, color = 'red')
    Xc, Yc, Zc = data_for_cylinder_along_z(0., 0., 140., 500)
    ax1.plot_surface(Xc, Zc, Yc, alpha = 0.15, color = "gray")
    # ax1.legend(bbox_to_anchor =(0.1,1))
    # plt.legend()
    fig.tight_layout(rect = (0., -0.2, 1., 1.1))
    plt.show()

def Plot_Cluster(cluster, number = ""):
    fig = plt.figure(dpi = 200)
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.view_init(None, 75)
    ax1.set_zlim((-140, 140))
    ax1.set_xlim((-140, 140))
    ax1.set_ylim((-0, 512))
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('t')
    # ax1.set_zlabel('Y')
    ax1.axis('off')
    # ax1.set_title("Before clustering: " + number)
    # ax1.scatter3D(raw_data[:, 0], raw_data[:, 2], raw_data[:, 1], marker = '.', s = 3, alpha = 1)    
    colors = ['b', 'green', 'orange', 'r', 'm', 'k', 'y', 'c', 'tan', 'salmon']
    #reta_centro = [[0, 0, 0], [0, 512, 0]]
    # ax1.quiver(0, 512, 0, -0, -512, -0, linestyle = 'solid', arrow_length_ratio = 0.05, lw = 1, color = 'red')
    i = 0
    for track in cluster:
        data = np.copy(track.get_data()[:, :3])
        direction = np.copy(track.get_versor())
        pb = track.get_pb()
        linepts = direction * np.mgrid[-800:800:2j][:, np.newaxis]
        linepts += pb
        x0, y0, z0, x1, y1, z1 = linepts[0][0], linepts[0][1], linepts[0][2], linepts[1][0], linepts[1][1], linepts[1][2]
        if track.isprimary:
            l = "PB"
        else:
            l = str(len(data))
        ax1.plot3D(np.array([x0, x1]),np.array([z0, z1]),np.array([y0, y1]), color = colors[i])        
        ax1.scatter3D(data[:, 0], data[:, 2], data[:, 1], marker = '.', s = 3, alpha = 1, c = colors[i])
        if track.vinchamber:
            vertex = track.get_vertex()
            ax1.scatter3D(vertex[0], vertex[2], vertex[1], marker = '^', s = 25, alpha = 0.85, c = colors[i])
        i += 1
    Xc, Yc, Zc = data_for_cylinder_along_z(0., 0., 280., 512)
    ax1.plot_surface(Xc, Zc, Yc, alpha=0.15, color = "gray")
    ax1.set_aspect("auto")
    fig.tight_layout(rect = (0., -0.2, 1., 1.1))
    # ax1.legend(bbox_to_anchor=(0.1,1))
    # plt.legend()
    plt.show()

def Plot_Cluster_ATTPC(cluster, number = ""):
    fig = plt.figure(dpi = 200)
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.view_init(None, 75)
    ax1.set_zlim((-260, 260))
    ax1.set_xlim((-260, 260))
    ax1.set_ylim((-200, 1200))
    ax1.set_xlabel('X')
    ax1.set_ylabel('t')
    ax1.set_zlabel('Y')
    ax1.axis('off')
    # ax1.set_title("Before clustering: " + number)
    # ax1.scatter3D(raw_data[:, 0], raw_data[:, 2], raw_data[:, 1], marker = '.', s = 3, alpha = 1)    
    colors = ['b', 'green', 'orange', 'r', 'm', 'k', 'y', 'c', 'tan', 'salmon']
    #reta_centro = [[0, 0, 0], [0, 512, 0]]
    ax1.quiver(0, 512, 0, -0, -512, -0, linestyle = 'solid', arrow_length_ratio = 0.05, lw = 1, color = 'red')
    i = 0
    for track in cluster:
        data = np.copy(track.get_data()[:, :3])
        direction = np.copy(track.get_versor())
        pb = track.get_pb()
        linepts = direction * np.mgrid[-800:800:2j][:, np.newaxis]
        linepts += pb
        x0, y0, z0, x1, y1, z1 = linepts[0][0], linepts[0][1], linepts[0][2], linepts[1][0], linepts[1][1], linepts[1][2]
        if track.isprimary:
            l = "PB"
        else:
            l = str(len(data))
        ax1.plot3D(np.array([x0, x1]),np.array([z0, z1]),np.array([y0, y1]), color = colors[i])        
        ax1.scatter3D(data[:, 0], data[:, 2], data[:, 1], marker = '.', s = 3, alpha = 1, c = colors[i])
        if track.vinchamber:
            vertex = track.get_vertex()
            ax1.scatter3D(vertex[0], vertex[2], vertex[1], marker = '^', s = 25, alpha = 0.85, c = colors[i])
        i += 1
    Xc, Yc, Zc = data_for_cylinder_along_z(0., 0., 280., 512)
    ax1.plot_surface(Xc, Zc, Yc, alpha=0.15, color = "gray")
    ax1.set_aspect("auto")
    fig.tight_layout(rect = (0., -0.2, 1., 1.1))
    # ax1.legend(bbox_to_anchor=(0.1,1))
    # plt.legend()
    plt.show()

def plot3dvs_ATTPC(before, after, number = "", name = ""):
    fig = plt.figure(figsize=plt.figaspect(0.4), dpi = 200)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    #ax1 = plt.axes(projection="3d")
    ax1.view_init(None, 75)
    ax2.view_init(None, 75)
    ax1.set_zlim((-250, 250))
    ax1.set_xlim((-250, 250))
    ax1.set_ylim((-200, 1200))
    ax2.set_zlim((-250, 250))
    ax2.set_xlim((-250, 250))
    ax2.set_ylim((-200, 1200))
    ax1.set_xlabel('X')
    ax1.set_ylabel('t')
    ax1.set_zlabel('Y')
    ax1.set_title("Before filter: " + number + " " + name)
    ax2.set_title("After filter: " + number + " " + name)
    ax2.set_xlabel('X')
    ax2.set_ylabel('t')
    ax2.set_zlabel('Y')
    ax1.scatter3D(before[:, 0], before[:, 2], before[:, 1], marker = '.', s = 3, alpha = 1, label = str(len(before)))
    ax2.scatter3D(after[:, 0], after[:, 2], after[:, 1], marker = '.', s = 3, alpha = 1, label = str(len(after)))
    plt.legend()
    plt.show()

def plot3dvs(before, after, number = "", name = ""):
    fig = plt.figure(figsize=plt.figaspect(0.4), dpi = 200)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    #ax1 = plt.axes(projection="3d")
    ax1.view_init(None, 75)
    ax2.view_init(None, 75)
    ax1.set_zlim((-140, 140))
    ax1.set_xlim((-140, 140))
    ax1.set_ylim((0, 512))
    ax2.set_zlim((-140, 140))
    ax2.set_xlim((-140, 140))
    ax2.set_ylim((0, 512))
    ax1.set_xlabel('X')
    ax1.set_ylabel('t')
    ax1.set_zlabel('Y')
    ax1.set_title("Before filter: " + number + " " + name)
    ax2.set_title("After filter: " + number + " " + name)
    ax2.set_xlabel('X')
    ax2.set_ylabel('t')
    ax2.set_zlabel('Y')
    ax1.scatter3D(before[:, 0], before[:, 2], before[:, 1], marker = '.', s = 3, alpha = 1, label = str(len(before)))
    ax2.scatter3D(after[:, 0], after[:, 2], after[:, 1], marker = '.', s = 3, alpha = 1, label = str(len(after)))
    plt.legend()
    plt.show()
    
def compare_ATTPC(raw_data, cluster, title = ""):
    fig = plt.figure(figsize=plt.figaspect(0.4), dpi = 200)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    #ax1 = plt.axes(projection="3d")
    ax1.view_init(None, 75)
    ax2.view_init(None, 75)
    for ax in(ax1, ax2):
        ax.set_zlim((-260, 260))
        ax.set_xlim((-260, 260))
        ax.set_ylim((-200, 1200))
        ax.set_xlabel('X')
        ax.set_ylabel('t')
        ax.set_zlabel('Y')
    ax1.set_title("Before filter: " + title)
    ax2.set_title("After filter: " + title)
    ax1.scatter3D(raw_data[:, 0], raw_data[:, 2], raw_data[:, 1], marker = '.', s = 3, alpha = 1)    
    i = 0
    colors = ['b', 'r', 'green', 'orange', 'm', 'k', 'y', 'c', 'tan', 'salmon']
    #reta_centro = [[0, 0, 0], [0, 512, 0]]
    ax1.quiver(0, 512, 0, -0, -512, -0, linestyle = 'solid', arrow_length_ratio = 0.05, lw = 1, color = 'red')
    ax2.quiver(0, 512, 0, -0, -512, -0, linestyle = 'solid', arrow_length_ratio = 0.05, lw = 1, color = 'red')
    for track in cluster:
        data = np.copy(track.get_data()[:, :3])
        direction = np.copy(track.get_versor())
        pb = track.get_pb()
        linepts = direction * np.mgrid[-800:800:2j][:, np.newaxis]
        linepts += pb
        x0, y0, z0, x1, y1, z1 = linepts[0][0], linepts[0][1], linepts[0][2], linepts[1][0], linepts[1][1], linepts[1][2]
        if track.isprimary:
            l = "PB"
        else:
            l = str(len(data))
        ax2.plot3D(np.array([x0, x1]),np.array([z0, z1]),np.array([y0, y1]), label = l, color = colors[i])        
        ax2.scatter3D(data[:, 0], data[:, 2], data[:, 1], marker = '.', s = 3, alpha = 1, c = colors[i])
        #print(track.get_vertex())
        if track.vinchamber:
            vertex = track.get_vertex()
            #print(vertex)
            ax2.scatter3D(vertex[0], vertex[2], vertex[1], marker = '^', s = 25, alpha = 0.85, c = colors[i])
        i += 1
    ax2.legend(bbox_to_anchor=(1.5,1))
    plt.legend()
    plt.show()
    #plt.close()
    
def compare(raw_data, cluster, title = ""):
    fig = plt.figure(figsize=plt.figaspect(0.4), dpi = 200)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    #ax1 = plt.axes(projection="3d")
    ax1.view_init(None, 75)
    ax2.view_init(None, 75)
    for ax in(ax1, ax2):
        ax.set_zlim((-140, 140))
        ax.set_xlim((-140, 140))
        ax.set_ylim((0, 512))
        ax.set_xlabel('X')
        ax.set_ylabel('t')
        ax.set_zlabel('Y')
    ax1.set_title("Before filter: " + title)
    ax2.set_title("After filter: " + title)
    ax1.scatter3D(raw_data[:, 0], raw_data[:, 2], raw_data[:, 1], marker = '.', s = 3, alpha = 1)    
    i = 0
    colors = ['b', 'r', 'green', 'orange', 'm', 'k', 'y', 'c', 'tan', 'salmon']
    #reta_centro = [[0, 0, 0], [0, 512, 0]]
    ax1.quiver(0, 512, 0, -0, -512, -0, linestyle = 'solid', arrow_length_ratio = 0.05, lw = 1, color = 'red')
    ax2.quiver(0, 512, 0, -0, -512, -0, linestyle = 'solid', arrow_length_ratio = 0.05, lw = 1, color = 'red')
    for track in cluster:
        data = np.copy(track.get_data()[:, :3])
        direction = np.copy(track.get_versor())
        pb = track.get_pb()
        linepts = direction * np.mgrid[-800:800:2j][:, np.newaxis]
        linepts += pb
        x0, y0, z0, x1, y1, z1 = linepts[0][0], linepts[0][1], linepts[0][2], linepts[1][0], linepts[1][1], linepts[1][2]
        if track.isprimary:
            l = "PB"
        else:
            l = str(len(data))
        ax2.plot3D(np.array([x0, x1]),np.array([z0, z1]),np.array([y0, y1]), label = l, color = colors[i])        
        ax2.scatter3D(data[:, 0], data[:, 2], data[:, 1], marker = '.', s = 3, alpha = 1, c = colors[i])
        #print(track.get_vertex())
        if track.vinchamber:
            vertex = track.get_vertex()
            #print(vertex)
            ax2.scatter3D(vertex[0], vertex[2], vertex[1], marker = '^', s = 25, alpha = 0.85, c = colors[i])
        i += 1
    ax2.legend(bbox_to_anchor=(1.5,1))
    plt.legend()
    plt.show()