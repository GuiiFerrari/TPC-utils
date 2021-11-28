# %matplotlib automatic
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 20:18:42 2020

@author: Guilherme Ferrari Fortino

Carrega e analisa dados (numpy)

"""

import numpy as np
import matplotlib.pyplot as plt
import uproot4 as up
from Analise import scattering_filter, scattering_filter2, make_data, get_number_tracks, get_inliers
from graph_functions import plot3dvs
import uproot4 as up
# import ROOT
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

plt.rcParams.update({'figure.max_open_warning': 0})

def plot_clusters(data, inliers, number = "", name = ""):
    outliers = np.array(list(set(range(len(data))) - set(inliers)))
    fig = plt.figure(figsize=plt.figaspect(0.4), dpi = 200)
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    #ax1 = plt.axes(projection="3d")
    ax1.view_init(None, 75)
    # ax2.view_init(None, 75)
    ax1.set_zlim((-250, 250))
    ax1.set_xlim((-250, 250))
    ax1.set_ylim((-300, 1300))
    # ax2.set_zlim((-250, 250))
    # ax2.set_xlim((-250, 250))
    # ax2.set_ylim((-300, 1300))
    ax1.set_xlabel('X')
    ax1.set_ylabel('t')
    ax1.set_zlabel('Y')
    ax1.set_title("Original")
    # ax2.set_title("After filter: " + number + " " + name)
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('t')
    # ax2.set_zlabel('Y')
    data_inliers  = data[inliers]
    data_outliers = data[outliers]
    ax1.scatter3D(data_inliers[:, 0], data_inliers[:, 2], data_inliers[:, 1], marker = '.', s = 14, alpha = 1, label = str(len(data)), c = "red")
    # if len(outliers > 0):
    ax1.scatter3D(data_outliers[:, 0], data_outliers[:, 2], data_outliers[:, 1], marker = '.', s = 1, alpha = 0.85, label = "Outliers", c = "black")
    plt.legend()
    plt.show()

# Runs abaixo
'''
#        0    1    2    3    4    5    6    7    8    9    10   11   12   13   14  15    16
runs = [143, 144, 145, 146, 147, 169, 170, 171, 172, 175, 189, 190, 191, 192, 193, 194, 195,
        196, 197, 198, 199, 200, 201, 202, 217]
#        17   18   19   20   21   22   23   24
'''
#num = runs[7]


# keys = ['TAC_T', 'TAC_A', 'Trigg_T', 'Trigg_A', 'IC_T', 'IC_A', 'Mesh_T', 'Mesh_A', 'alpha_T', 'alpha_A']
# graphs = [["alpha_A", "alpha_T"], ["alpha_T", "Mesh_A"], ["alpha_A", "Mesh_A"],
#           ["Mesh_T", "Mesh_A"], ["alpha_A", "Mesh_T"], ["IC_A", "alpha_T"],
#           ["Mesh_T", "IC_A"], ["IC_A", "Mesh_A"], ["alpha_A", "IC_A"], ["IC_T", "IC_A"],
#           ["IC_T", "Mesh_A"], ["alpha_A", "IC_T"]]
# graphs_2 = [["TAC_A", "alpha_A"], ["TAC_A", "Mesh_A"]]

# lims = {"alpha_T" : 512, "Mesh_T" : 512, "alpha_A" : 4000, "Mesh_A" : 4000, "IC_A" : 50, "IC_T" : 512}
# num_bins = {"alpha_T" : 50, "Mesh_T" : 50, "IC_T" : 50, "Mesh_A" : 400, "IC_A" : 20, "alpha_A" : 400, "TAC_A" : 560}

def DA():
    parcial = [220] # [192] # 192 pro 14O
    smp = [24]  #[15, 20, 30, 35, 40]
    d   = [15.] #[10, 15, 20, 25]

    for num in parcial:
        # Pega o conteúdo do arquivo.
        arquivo = './output_run_%d_digi.root'%num
        # arquivo = './Cloudrun_0085.root'
        #branches = None
        f = up.open(arquivo)
        trees = f.keys()
        print("\n\nnum = ", num, "\nTrees = ", trees)
        
        # eventos  = list()
        # cleans   = list()
        # titulo   = list()
        tree = trees[0]
        branches = f[tree].arrays(library = "np")
        f.close()
        # Aplicando todos os filtros
        # aaa = 32
        i = 0
        # part = 0q
        eventos = list()
        print("Tamanho total = ", len(branches["vpx"]), "\n")
        # number_events_raw = range(len(branches["vpx"]))
        number_events = list()
        # eventos_uteis = np.load("eventos_filtro.npy", allow_pickle=True)
        # print(eventos_uteis)
        # print(len(branches["Trigg_A"][2]))
        # a = np.random.choice(range(len(branches["vpx"])), 150)
        # for i in a: #len(branches["vpx"])):
        for i in range(len(branches["vpx"])): # 34, 35, 49
        # for i in range(8, len(branches["vpx"])):
        # for i in range(41, len(branches["vpx"])):
        # for i in range(41, 42):
            if len(branches["TAC_A"][i]) > 1:
                if branches["TAC_A"][i][0] < 500:
                    evento = np.dstack((branches["vpx"][i], branches["vpy"][i], branches["vpt"][i], branches["vpe"][i]))[0]
                    eventos.append(evento.astype(float))
                    number_events.append(i)
            else:
                if branches["TAC_A"][i] < 500 and len(branches["vpx"][i]) >= smp[0]:
                    evento = np.dstack((branches["vpx"][i], branches["vpy"][i], branches["vpt"][i], branches["vpe"][i]))[0]
                    eventos.append(evento.astype(float))
                    number_events.append(i)
            # if len(branches["vpx"][i]) >= smp[0]:
            #     evento = np.dstack((branches["vpx"][i], branches["vpy"][i], branches["vpz"][i], branches["vpe"][i]))[0]
            #     eventos.append(evento.astype(float))
            #     number_events.append(i)
        # pulsos = []
        # eventos_uteis = np.array([ 924, 2523, 4563, 4994, 5415], dtype = int)
        # eventos_uteis = np.array([ 924, 4563, 4994, 5415], dtype = int)
        # np.save("eventos_filtro.npy", eventos_uteis)
        # for i in eventos_uteis:
            # evento = np.dstack((branches["vpx"][i], branches["vpy"][i], branches["vpz"][i], branches["vpe"][i]))[0]
            # print(branches["vpt"][i])
            # pulsos.append(np.array(branches["pulses"][i].tolist(), dtype = float))
            # eventos.append(evento.astype(float))
        #     number_events.append(i)
        # pulsos = np.array(pulsos, dtype = object)
        # np.save("pulsos_filtro.npy", pulsos)
        # np.save("eventos_filtro.npy", np.array(eventos, dtype = object))
        # index = [1396, 2672, 2945, 3135, 3179, 3245, 3416, 4163, 4510, 5857, 5877, 6996, 8902, 8957, 9684, 10241, 10454, 11015, 11957, 12053, 12436, 15479, 16009, 16659, 17202,
        #      17537, 17936, 18350, 18356, 18487, 18722, 19263, 19295, 20155, 20511, 22217, 23346, 24343, 24700, 24835, 25414, 26000, 27815, 28090, 28336, 28391, 28721, 28807,
        #      29096, 29222]
        # eventos = eventos[index]
        # number_events = number_events[index]
            
        parcial = scattering_filter(events = eventos, number_events = number_events, 
                                    run = num, ve_filter = 110.0,
                                    search_radius = 20, 
                                    min_neigh = 3, samples = smp[0],
                                    dist = d[0], th = 1.75*d[0], inner = 9.,
                                    verb = False, part = 5, It = 800, 
                                    mode = 3, diametro = 30., save = False,
                                    comp = False, logger = False,
                                    branches = branches,
                                    all_events = range(len(branches["vpx"])))
        # parcial = get_number_tracks(events = eventos, number_events = number_events, 
        #                             run = num, ve_filter = 110.0,
        #                             search_radius = 20, 
        #                             min_neigh = 3, samples = smp[0],
        #                             dist = d[0], th = 1.75*d[0], inner = 9.,
        #                             verb = False, part = 5, It = 800, 
        #                             mode = 3, diametro = 30., save = False,
        #                             comp = False, logger = False,
        #                             branches = branches,
        #                             all_events = range(len(branches["vpx"])))
        # make_data(events = eventos, number_events = number_events,
        #           run = num, ve_filter = 110.0,
        #           search_radius = 20, min_neigh = 3, samples = smp[0],
        #           all_events = range(len(branches["vpx"])), branches = branches)

def load_utils():
    eventos_uteis = np.load("eventos_filtro.npy", allow_pickle=True)
    pulsos        = np.load("pulsos_filtro.npy", allow_pickle=True)
    print(len(eventos_uteis))
    print(len(pulsos))
    get_inliers(eventos_uteis)

def load_inliers():
    eventos_uteis = np.load("eventos_filtro.npy", allow_pickle=True)
    pulsos        = np.load("pulsos_filtro.npy", allow_pickle=True)
    inliers       = np.load("inliers_filtro.npy", allow_pickle=True)
    tent_1 = np.array([0, 2, 3, 4], dtype = int)
    eventos_uteis = eventos_uteis[tent_1]
    pulsos        = pulsos[tent_1]
    # inliers       = inliers[tent_1]
    print(len(eventos_uteis))
    print(len(pulsos))
    print(len(inliers))
    for data, inlier in zip(eventos_uteis, inliers):
        print(data.shape, inlier.shape)
        plot_clusters(data, inlier)

def plot_pulses(pulso):
    xt = np.arange(0.5, 512, 1)
    plt.figure(dpi = 200)
    plt.plot(xt, pulso, lw = 2.)
    plt.xlim(0, 512)
    plt.show()

def load_pulses():
    eventos_uteis = np.load("eventos_filtro.npy", allow_pickle = True)
    pulsos        = np.load("pulsos_filtro.npy",  allow_pickle = True)
    inliers       = np.load("inliers_filtro.npy", allow_pickle = True)
    tent_1 = np.array([0, 2, 3, 4], dtype = int)
    eventos_uteis = eventos_uteis[tent_1]
    pulsos        = pulsos[tent_1]
    # inliers       = inliers[tent_1]
    print(len(eventos_uteis))
    print(len(pulsos))
    print(len(inliers))
    for evento, pulsos, inlier in zip(eventos_uteis, pulsos, inliers):
        outliers = np.array(list(set(range(len(evento))) - set(inlier)))
        for i in outliers:
            plot_pulses(pulsos[i])

def load_ml_events():
    # dados = np.load("nuvens_ml.npy", allow_pickle = True)
    dados = np.load("nuvens_ml_3.npy", allow_pickle = True)
    # dados = [np.load("Evento_1950.npy", allow_pickle = True)]
    # dados = [np.load("Evento_1566.npy", allow_pickle = True)]
    # print(dados[0].shape)
    result = get_inliers(dados, ve_filter = 1.)
    np.save("results_ml_4.npy", result)
    print(result[0].shape)
    # total_charges = []
    # total_angles  = []
    # total_vertex  = []
    # # total_track   = []
    # total_length  = []
    # total_points  = []
    # total_info    = []
    # for cluster in result:
    #     for track in cluster:
    #         # if track.isprimary == False:
    #         if track.isprimary == False:
    #             total_charges.append(track.TCharge)
    #             total_angles.append(track.angle)
    #             total_vertex.append(track.vertex[2])
    #             total_length.append(track.length)
    #             total_points.append(len(track.data))
    #             if track.isprimary == False:
    #                 total_info.append(0)
    #             else:
    #                 total_info.append(1)
    #         # if track.TCharge/len(track.data) > 1600.:
    #         #     compare_clustering(cluster, cluster)
    #             # total_track.append(track.length)
    # total_charges = np.array(total_charges, dtype = float)
    # total_angles  = np.array(total_angles, dtype = float)
    # total_vertex  = np.array(total_vertex, dtype = float)
    # # total_track   = np.array(total_track, dtype = float)
    # total_length  = np.array(total_length, dtype = float)
    # total_points  = np.array(total_points, dtype = int)
    # total_info    = np.array(total_info, dtype = int)
    # np.savez_compressed("results_ml_model.npz", total_charges = total_charges,
    #  total_angles = total_angles, total_vertex = total_vertex, total_length = total_length,
    #  total_points = total_points, total_info = total_info)


if __name__ == "__main__":
    # get_pulses()
    # DA()
    load_ml_events()
    # load_utils()
    # load_inliers()
    # load_pulses()

