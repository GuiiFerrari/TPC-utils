import numpy as np
import matplotlib.pyplot as plt
import uproot4 as up
# from Analise import scattering_filter_clustering
import hdbscan as hdb
from track import Track, get_versor_cpp
from Analise import compare_clustering, clustering_algo, redefine_vertex_tracks, select_tracks, Plot_Cluster_raw, o3d_outlier_removal, plot3dvs
from time import perf_counter

def plot_clusters(data, clusters, outliers, number = "", name = ""):
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
    ax1.set_title("Original")
    ax2.set_title("After filter: " + number + " " + name)
    ax2.set_xlabel('X')
    ax2.set_ylabel('t')
    ax2.set_zlabel('Y')
    ax1.scatter3D(data[:, 0], data[:, 2], data[:, 1], marker = '.', s = 1, alpha = 1, label = str(len(data)))
    if len(outliers > 0):
        ax2.scatter3D(outliers[:, 0], outliers[:, 2], outliers[:, 1], marker = '.', s = 1, alpha = 0.85, label = "Outliers", c = "black")
    for i in range(len(clusters)):
        ax2.scatter3D(clusters[i][:, 0], clusters[i][:, 2], clusters[i][:, 1], marker = '.', s = 5, alpha = 1, label = str(i))
    plt.legend()
    plt.show()

def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    z = np.linspace(0, height_z, 150)
    theta = np.linspace(0, 2*np.pi, 150)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

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
    Xc, Yc, Zc = data_for_cylinder_along_z(0., 0., 140, 512)
    ax1.plot_surface(Xc, Zc, Yc, alpha=0.15, color = "gray")
    ax1.set_aspect("auto")
    # fig.tight_layout(rect = (0., -0.2, 1., 1.1))
    fig.tight_layout()
    # ax1.legend(bbox_to_anchor=(0.1,1))
    # plt.legend()
    plt.show()

def update_bar(percent, tam_bar, end = "\r"):
    num_chars = int(percent*tam_bar/100)
    print("[" + "#"*num_chars + " "*(tam_bar - num_chars) + f"] {percent:.2f}%", end = end)


def fit_point_cloud(X):
    """
    Computa os valores de outliers para x e compara com a previsão y.
    """
    algo   = hdb.HDBSCAN(min_cluster_size = 12, min_samples = 5, cluster_selection_epsilon = 0.,
                        metric = "euclidean", leaf_size = 40, core_dist_n_jobs = 10, allow_single_cluster = True,
                        approx_min_span_tree = True).fit(X)
    labels = algo.labels_
    return algo, labels

def get_clusters(data, labels):
    max_label = labels.max()
    clusters  = [data[np.where(labels == i)[0]] for i in range(max_label + 1)]
    outliers  = data[np.where(labels == -1)]
    return clusters, outliers

if __name__ == "__main__":
    d        = 15.*1.75
    smp      = 24
    num      = 220
    f        = up.open('./output_run_%d_digi.root'%num)
    branches = f[f.keys()[0]].arrays(library = "np")
    eventos       = list()
    number_events = list()
    print("Tamanho total = ", len(branches["vpx"]), "\n")
    for i in range(0, len(branches["vpx"])):
        if len(branches["TAC_A"][i]) > 1:
            if branches["TAC_A"][i][0] < 500:
                evento = np.dstack((branches["vpx"][i], branches["vpy"][i], branches["vpt"][i], branches["vpe"][i]))[0]
                eventos.append(evento.astype(float))
                number_events.append(i)
        else:
            if branches["TAC_A"][i] < 500 and len(branches["vpx"][i]) >= smp and len(branches["vpx"][i]) < 500:
                evento = np.dstack((branches["vpx"][i], branches["vpy"][i], branches["vpt"][i], branches["vpe"][i]))[0]
                eventos.append(evento.astype(float))
                number_events.append(i)
    print(len(eventos))
    index = [1396, 2672, 2945, 3135, 3179, 3245, 3416, 4163, 4510, 5857, 5877, 6996, 8902, 8957, 9684, 10241, 10454, 11015, 11957, 12053, 12436, 15479, 16009, 16659, 17202,
             17537, 17936, 18350, 18356, 18487, 18722, 19263, 19295, 20155, 20511, 22217, 23346, 24343, 24700, 24835, 25414, 26000, 27815, 28090, 28336, 28391, 28721, 28807,
             29096, 29222]

    t_i = perf_counter()
    # result = []
    # for i in range(len(eventos)):
    # # for numero, i in enumerate(index):
    #     data = eventos[i]
    #     # Plot_Cluster_raw(data)
    #     data = data[np.where(data[:, 3] >= 110.0)[0]]
    #     pc1 = o3d_outlier_removal(data, 3, 12)
    #     # plot3dvs(data, pc1)
    #     _, labels = fit_point_cloud(X = pc1[:, :3])
    #     clusters, outliers = get_clusters(pc1, labels)
    #     # clusters, outliers = get_clusters(data, labels)
    #     # plot_clusters(pc1, clusters, outliers)
    #     pc2 = []
    #     for cluster in clusters:
    #         versor, pb = get_versor_cpp(cluster)
    #         pc2.append(Track(cluster, versor, pb))
        
    #     pc3 = clustering_algo(pc2, d, 9.)
    #     result.append(pc3)
    result = []
    tam = len(eventos)
    tam_bar = 50
    for i in range(len(eventos)):
        update_bar(i*100/tam, tam_bar)
    # for numero, i in enumerate(index):
        data = eventos[i]
        # Plot_Cluster_raw(data)
        data = data[np.where(data[:, 3] >= 110.0)[0]]
        pc1 = o3d_outlier_removal(data, 3, 12)
        if len(pc1) >= smp:
            _, labels = fit_point_cloud(X = pc1[:, :3])
            clusters, outliers = get_clusters(pc1, labels)
            # clusters, outliers = get_clusters(data, labels)
            # plot_clusters(pc1, clusters, outliers)
            pc2 = []
            for cluster in clusters:
                versor, pb = get_versor_cpp(cluster)
                pc2.append(Track(cluster, versor, pb))
            if len(pc2) > 1:
                pc3 = clustering_algo(pc2, d, 9.)
                # Plot_Cluster(pc3)
                if len(pc3) > 1:
                    pc4 = redefine_vertex_tracks(pc3)
                    pc5 = select_tracks(pc4, 30.)
                    if len(pc5) > 1:
                        for track in pc5:
                            track.calculate_length()
                        result.append(pc5)
                    elif len(pc5) == 1:
                        if pc5[0].isprimary == False and pc5[0].vinchamber == True and pc5[0].get_min_dist() <= 30.:
                            pc5[0].calculate_length()
                            result.append(pc5)
                elif len(pc3) == 1:
                    if pc3[0].isprimary == False and pc3[0].vinchamber == True and pc3[0].get_min_dist() <= 30.:
                        pc3[0].calculate_length()
                        result.append(pc3)
            elif len(pc2) == 1:
                if pc2[0].isprimary == False and pc2[0].vinchamber == True and pc2[0].get_min_dist() <= 30.:
                    pc2[0].calculate_length()
                    result.append(pc2)
    update_bar(100, tam_bar, end = "\n")
    t_f = perf_counter() - t_i
    print(f"Tempo de execução para {len(eventos)} eventos = {t_f:.2f} s = {t_f/60:.2} min.")
    print(f"Eficiência  = {len(result)*100/len(eventos)} %")
        # compare_clustering(pc2, pc3, f"{i}")
        # Plot_Cluster(pc3)
        # compare_clustering(pc1, pc2, number = f"{i} numero = {numero}")
        # pc3 = redefine_vertex_tracks(pc2)
        # pc4 = select_tracks(pc3)
        # compare_clustering(pc3, pc4, number = f"{i} numero = {numero}")
        # compare_clustering(pc2, pc3, number = f"{i}")
        # compare_clustering(pc1, pc2, number = f"{i}")

        # plot_clusters(data, clusters, outliers, number = f"{i}", name = "")
    
            