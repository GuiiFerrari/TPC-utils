# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:10:05 2020

@author: guilh

Faz análises de clusters

"""

import numpy as np
from numpy import degrees, arccos, array, cross
from graph_functions import compare, compare_ATTPC, compare_clustering_ATTPC, compare_clustering, plot3dvs, plot3dvs_ATTPC
from graph_functions import Plot_Cluster_ATTPC, Plot_Cluster, Plot_Cluster_raw, Plot_Cluster_raw_ATTPC, plot_4DLine
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm
from sklearn.metrics import silhouette_score
#from scipy.spatial.distance import pdist
from Line_algorithms import Line6, Line7, get_info
from time import time
#from utils.utils import Line
import open3d as o3d
from datetime import datetime
#from fit3d import Line
from copy import deepcopy
# import sys
# import concurrent.futures as cf
# import ROOT
# from ctypes import c_double
# from sys import getsizeof
# from pympler import asizeof
# from ROOT import *

plt.rcParams.update({'figure.max_open_warning': 0})

def get_4inliersPY(data, line_algo, min_samples: int = 30):
    '''
    Return a cluster which has tracks.

    Parameters
    ----------
    data : np.ndarray
        4-vector dataset to be analyzed.
    min_samples : int, optional
        Minimum number of points to be considered a track. The default is 30.
    min_dist : float, optional
        Minimum distance of a point to be considered as part of the track. The default is 5.
    it : int, optional
        Number of iterations. The default is 1000.
    mode : int, optional
        Mode of the random sampling. The default is 0.

    Returns
    -------
    clusters : list
        List which has multiple tracks detecteds.

    '''
    clusters = line_algo.fit(data)
    # x, y, t, e = np.copy(data[:, 0]), np.copy(data[:, 1]), np.copy(data[:, 2]), np.copy(data[:, 3])
    # pcloud     = np.array([x, y, t], dtype = float).T
    # clusters   = list()  
    # while True:
    #     #line = Line2(pcloud, it, min_dist, e, mode = mode, selection_mode = 1)
    #     inliers, versor, pb = line_algo.fit(pcloud, e)
    #     if len(inliers) < min_samples:
    #         break
    #     res = np.dstack((np.copy(x[inliers]), np.copy(y[inliers]), np.copy(t[inliers]), np.copy(e[inliers])))[0]
    #     new_track = Track(res, versor, pb)
    #     clusters.append(new_track)
    #     x, y, t, e = np.delete(x, inliers), np.delete(y, inliers), np.delete(t, inliers), np.delete(e, inliers)
    #     if len(x) < min_samples:
    #         break
    #     pcloud = np.array([x, y, t], dtype = float).T    
    # outliers = np.vstack((x, np.vstack((y, np.vstack((t, e)))))).T
    return clusters

def o3d_outlier_removal(data, min_points, search_radius):
    if len(data) < 1:
        return []
    pcloud = o3d.geometry.PointCloud()
    pcloud.points = o3d.utility.Vector3dVector(np.copy(data[:, :3]))
    _, inliers = pcloud.remove_radius_outlier(nb_points = min_points, radius = search_radius)
    return np.copy(data[inliers]).astype(float)

def o3d_outlier_removal_2(data, extra_info, min_points, search_radius):
    if len(data) < 1:
        return []
    pcloud = o3d.geometry.PointCloud()
    pcloud.points = o3d.utility.Vector3dVector(np.copy(data[:, :3]))
    _, inliers = pcloud.remove_radius_outlier(nb_points = min_points, radius = search_radius)
    return np.copy(data[inliers]).astype(float), extra_info[inliers]
    
def regroup(cluster, uniques):
    final  = list()
    i, j   = uniques
    # print(i, j, len(cluster))
    new_track = cluster[i] + cluster[j] 
    final.append(new_track)
    indices = [z for z in range(len(cluster))]
    indices.remove(i)
    indices.remove(j)
    for k in indices:
        final.append(cluster[k])
    return final

def get_distance(track1, track2):
    if get_angle(track1.get_versor(), track2.get_versor()) == 0.:
        # print(np.abs(norm(track1.get_pb() - track2.get_pb())))
        return norm(cross(track1.get_versor(), track1.get_pb() - track2.get_pb()))
    Vc = cross(track1.get_versor(), track2.get_versor())
    Vc = Vc/norm(Vc)
    return np.abs(np.dot(Vc, track1.get_pb() - track2.get_pb()))

#@njit(parallel=True)
def get_angle(versor1, versor2):
    inner = np.inner(versor1, versor2)
    return degrees(arccos(np.abs(inner)))

def point_line_dists(points, versor, pb):
    # versor_stack =  np.stack([versor]*len(points),0)
    normas = norm(cross(np.stack([versor]*len(points),0), pb - points), axis = 1)
    return normas

#@njit(parallel=True)
def get_inffo(clusters, coefs, threshold = 0.1):
    if len(clusters) <= 1: return None
    uniques = list()
    for i in range(len(clusters) - 1):
        for j in range(len(clusters) - 1, i, -1):
            ci = np.copy(clusters[i])
            cj = np.copy(clusters[j])
            versor1, versor2 = np.copy(coefs[i]), np.copy(coefs[j])
            X = np.vstack((ci, cj))
            li, lj = array([0]*len(ci)), array([1]*len(cj))
            L = np.hstack((li, lj))
            m = silhouette_score(X, L)
            uniques.append([m, get_angle(versor1, versor2)])
    u = np.array(uniques[0], ndmin = 2)
    for i in range(1, len(uniques)):
        u = np.vstack((u, uniques[i]))
    return u

def get_uniques(cluster, threshold = 10., inner_angle:float  = 10.0):
    # return []
    if len(cluster) < 2: return []
    # print("\n")
    for ii in range(len(cluster) - 1):
        for jj in range(len(cluster) - 1, ii, -1):
            versor1, versor2 = cluster[ii].get_versor(), cluster[jj].get_versor()
            angulo1 = cluster[ii].angle # np.abs(get_angle(versor1, np.array([0., 0., 1.])))
            angulo2 = cluster[jj].angle # np.abs(get_angle(versor2, np.array([0., 0., 1.])))
            m       = np.abs(angulo1 - angulo2)
            # print("Ângulo entre as retas %d e %d = %.3f"%(ii, jj, m))
            if m < inner_angle:
                a        = cluster[jj].get_data()[:, :3]
                pb       = cluster[ii].get_pb()
                # dist_med = np.sum(point_line_dists(a, versor1, pb))/len(a)
                dist_med = np.sum(norm(cross(np.stack([versor1]*len(a),0), pb - a), axis = 1))/len(a)
                # print("Distância 1 media reta %d %d: %.5f"%(ii, jj, dist_med))
                # print("Distância entre duas retas %d e %d: %.2f"%(i, j, distancia))
                if dist_med < threshold:
                    return [ii, jj]
    return []
    
def get_vertex(track):
    primary = np.array([0., 0., -1.], dtype = float)
    versor  = track.get_versor()
    p1, p2  = track.get_pb(), np.array([0., 0., 512.0], dtype = float)
    Vc      = cross(versor, primary)
    Vc      = Vc/norm(Vc)
    min_distance = np.abs(np.dot(Vc, p1 - p2))
    RHS = p2 - p1
    LHS = array([versor, -primary, Vc]).T
    t1, t2, _ = np.linalg.solve(LHS, RHS)
    point_line1 = p1 + t1*versor
    point_line2 = p2 + t2*primary
    point = 0.5*(point_line1 + point_line2)
    return point, min_distance

def correct_cluster(old_cluster, time):
    ind = np.where(old_cluster[:, 2] > time)[0]
    #print(old_cluster.shape)
    if len(ind) > 6 and len(ind) < len(old_cluster) - 6:
        before_reac = np.copy(old_cluster[ind])
        mask        = np.ones(old_cluster.shape[0], dtype=bool)
        mask[ind]   = False
        after_reac  = np.copy(old_cluster[mask])
        #after_reac  = np.delete(after_reac, ind).reshape((-1, 4))
        #print(after_reac.shape, before_reac.shape)
        # before_versor, after_versor = get_4versor(before_reac), get_4versor(after_reac)
        #print(len(ind), len(before_reac), len(after_reac))
        # return before_reac, before_versor, after_reac, after_versor
    return None, None, None, None

def clustering_algo(cluster, threshold: float  = 0.1, inner_angle: float = 15.0):
    # print("len entrada = ", len(cluster))
    uniques = get_uniques(cluster, threshold = threshold, inner_angle = inner_angle) 
    # print("len saída   = ", len(cluster))   
    while(len(uniques) > 0):
        # print("len while   = ", len(cluster))
        cluster = regroup(cluster, uniques)
        uniques = get_uniques(cluster, threshold = threshold, inner_angle = inner_angle)
    return cluster

def redefine_vertex_tracks(cluster):
    has_p = False
    # print("Antes = ", len(cluster))
    for i in range(len(cluster)):
        if cluster[i].isprimary == True:
            track   = cluster.pop(i)
            cluster.insert(0, track)
            has_p   = True
            break
    if has_p == False:
        return cluster
    if len(cluster) == 2:
        cluster[1].redefine_vertex(cluster[0])
        return cluster
    for i in range(1, len(cluster)):
        cluster[i].redefine_vertex(cluster[0])
    return cluster

def select_tracks(cluster, diametro = 30.):
    new_cluster = list()
    for i in range(len(cluster)):
        if cluster[i].isprimary == False:
            if cluster[i].vinchamber == True and cluster[i].get_min_dist() < diametro:
                new_cluster.append(cluster[i])
        else:
            new_cluster.append(cluster[i])
    return new_cluster

#@ray.remote
def scattering_filter(events, number_events, run, ve_filter = 100., search_radius = 12., min_neigh = 4, samples = 30, dist = 10, th = 15., inner = 15., verb = False, part = 0, It = 1000, **kwargs):
    """
    Receive the data and select only the ones with scattering. Divided in three steps:
    First:  Select points with only a minimum energy.
    Second: Use outlier removal to clean more outliers.
    Third:  Applies ransac with clustering to get the trajectories and select cluster with two or more tracks.

    Parameters
    ----------

    events : list
        Set of all data to be filtered.
    run : int
        Number of the run.
    ve_filter : int or float, optional
        Energy filter. The default is 100.
    search_radius : int or float, optional
        Search radius for outlier removal. The default is 12.
    min_neigh : int, optional
        Minimum of neighbors for outlier removal. The default is 4.
    samples : int, optional
        Parameter of ransac. The default is 30.
    dist : int, optional
        Parameter of ransac. The default is 10.
    th : float, optional
        Threshold, parameter for clustering after ransac. The default is 10.
    inner : float, optional
        Parameter for clustering after ransac. The default is 15.
    verb : bool, optional
        If True plto all the scattering events. The default is False.
    part : int, optional
        Parameter to save files. The default is 0.
    It : int, optional
        Number of ransac iterations. The default is 1000.

    Returns
    -------
    None.

    """
    # save       = kwargs.pop('save', False)
    comp       = kwargs.pop('comp', False)
    mode       = kwargs.pop('mode', 1)
    logger     = kwargs.pop('logger', True)
    diam       = kwargs.pop('diametro', 30.)
    # branches   = kwargs.pop('branches', None)
    # all_events = kwargs.pop("all_events", [])
    # pulsos     = kwargs.pop("pulsos", [])
    line_algo  = Line7(It, dist, samples, mode)
    # print("Tamanho = ", asizeof.asizeof(line_algo))

    s1 = "\n\tStarting the Scattering search.\n\n"
    
    print(s1)
    
    param = "\nve_filter = %f | search_radius = %f | min_neigh = %d | samples = %d | "%(ve_filter, search_radius, min_neigh, samples)
    param += "dist = %d | th = %f | inner = %f | It = %d"%(dist, th, inner, It)
    # number_events = range(len(events))
    
    s2 = "\nInformations:\nRun number %d\nNumber of events: %d\n\n"%(run, len(number_events))
    print(s2)
    
    # Apllying the energy filter

    initial_time = time()
    s3 = "Starting the search."
    print(s3)
    result       = list()
    number_event = list()
    # index2 = [1396, 2672, 2945, 3135, 3179, 3245, 3416, 4163, 4510, 5857, 5877, 6996, 8902, 8957, 9684, 10241, 10454, 11015, 11957, 12053, 12436, 15479, 16009, 16659, 17202,
    #          17537, 17936, 18350, 18356, 18487, 18722, 19263, 19295, 20155, 20511, 22217, 23346, 24343, 24700, 24835, 25414, 26000, 27815, 28090, 28336, 28391, 28721, 28807,
    #          29096, 29222]
    # eventos_filtro = list()
    MAX_EVENTS = len(number_events)
    for indice1, (data, number) in enumerate(zip(events, number_events)):
        update_bar(indice1*100/MAX_EVENTS, 30)
    # for i in index2:
        # data = events[i]
        # number = number_events[i]
        # print("Análise evento ", number)
        # Plot_Cluster_raw(data, number = number)
        # Energy filter
        index = np.where(data[:, 3] >= ve_filter)[0]
        # print("Filtro de energia ok")
        if len(index) >= samples:#:
            # Outlier removal
            aux = data[index]
            # Abaixo opção de plot vs
            # plot3dvs(data, aux)
            new_cluster = o3d_outlier_removal(aux, min_points = min_neigh, search_radius = search_radius)
            # print("Outlier removal ok")
            # Abaixo opção de plot vs
            plot3dvs(data, new_cluster)
            # print("Tamanho do cluster de número %d: %d"%(number, len(new_cluster)))
            if len(new_cluster) >= samples:
                # Ransac + clustering
                cluster = get_4inliersPY(new_cluster, line_algo, min_samples=samples)
                # Plot_Cluster(cluster)
                # compare(new_cluster, cluster, title = str(number))
                # print("Ransac ok")
                if len(cluster) > 1:
                    cluster2 = clustering_algo(deepcopy(cluster), th, inner)
                    # print(f"Número de tracks = {len(cluster2)}")
                    # print("Tamanho  = ", asizeof.asizeof(cluster2))
                    # for aaa in cluster2:
                        # print("Tamanho  = ", asizeof.asizeof(aaa))
                    # Plot_Cluster(cluster2)
                    compare_clustering(cluster, cluster2, number = str(number))
                    # if len(new_cluster) < samples:
                    #     compare(data, cluster2, title = str(number) + "Não deu certo")
                    # else:
                    #     compare(data, cluster2, title = str(number))
                    if 1 < len(cluster2):
                        cluster3 = redefine_vertex_tracks(deepcopy(cluster2))
                        # compare_clustering(cluster2, cluster3, number = str(number))
                        # result.append(cluster3)
                        # # result.append(cluster2)
                        # number_event.append(number)
                        cluster4 = select_tracks(cluster3, diametro = diam)
                        # compare_clustering(cluster3, cluster4, number = str(number) + f" {i}")
                        # compare(new_cluster, cluster4, title = str(number) + f" {i}")
                        # partial_inliers = [] 
                        # for track in cluster4:
                        #     partial_inliers.append(np.where((data[:, 0] == track.data[:, 0]) & (data[:, 1] == track.data[:, 1]) & (data[:, 2] == track.data[:, 2]) & (data[:, 3] == track.data[:, 3])))
                        # inliers_EV.append(partial_inliers)
                        # print(partial_inliers)
                        # print(cluster[0].inliers)
                        # for track in cluster4:
                        #     if track.TCharge/len(track.data) > 1600.:
                        #         # compare(data, cluster4)
                        #         eventos_filtro.append(number)
                        #         break

                        if len(cluster4) > 1:
                            # compare_clustering(cluster3, cluster4, number = str(number))
                            # for reta in cluster4:
                            #     reta.calculate_length()
                            #     reta.set_inliers(data)
                            # print("Evento %d aceito"%number)
                            # if 350 <= cluster4[0].vertex[2] <= 400:
                            #     Plot_Cluster(cluster4, number = number)
                            result.append(cluster4)
                            number_event.append(number)
                        elif len(cluster4) == 1 and cluster4[0].isprimary == False:
                            # print("Evento %d aceito"%number)
                            cluster4[0].calculate_length()
                            # cluster[0].set_inliers(data)
                            # compare_clustering(cluster3, cluster4, number = str(number))
                            result.append(cluster4)
                            number_event.append(number)

                    elif len(cluster2) == 1:
                        if cluster2[0].isprimary == False and cluster[0].vinchamber == True and cluster[0].get_min_dist() <= diam:
                            # print("Evento %d aceito"%number)
                            cluster2[0].calculate_length()
                            # partial_inliers = [] 
                            # for track in cluster4:
                            #     partial_inliers.append(np.where((data[:, 0] == track.data[:, 0]) & (data[:, 1] == track.data[:, 1]) & (data[:, 2] == track.data[:, 2]) & (data[:, 3] == track.data[:, 3])))
                            # inliers_EV.append(partial_inliers)
                            # print(partial_inliers)
                            # print(cluster[0].inliers)
                            # for track in cluster2:
                            #     if track.TCharge/len(track.data) > 1600.:
                            #         # compare(data, cluster2)
                            #         eventos_filtro.append(number)
                            #         break
                            # cluster[0].set_inliers(data)
                            result.append(cluster2)
                            number_event.append(number)
                elif len(cluster) == 1:
                    if cluster[0].isprimary == False and cluster[0].vinchamber == True and cluster[0].get_min_dist() <= diam:
                        # print("Evento %d aceito"%number)
                        cluster[0].calculate_length()
                        # partial_inliers = [] 
                        # for track in cluster4:
                        #     partial_inliers.append(np.where((data[:, 0] == track.data[:, 0]) & (data[:, 1] == track.data[:, 1]) & (data[:, 2] == track.data[:, 2]) & (data[:, 3] == track.data[:, 3])))
                        # inliers_EV.append(partial_inliers)
                        # print(partial_inliers)
                        # print(cluster[0].inliers)
                        # for track in cluster:
                        #         if track.TCharge/len(track.data) > 1600.:
                        #             # compare(data, cluster)
                        #             eventos_filtro.append(number)
                        #             break
                        # cluster[0].set_inliers(data)
                        result.append(cluster)
                        number_event.append(number)
                # else:
                #     compare(data, cluster2, title = str(number) + "Não deu certo: " + str(len(data)))

    final_time = time()
    update_bar(100, 30, end = "\n")
    # eventos_filtro = np.array(eventos_filtro, dtype = int)
    # np.save("eventos_filtro.npy", eventos_filtro)
            # final_data.append(np.array(aux))
        # np.save("scattering_17F_events_%d_%d"%(run, part), np.array(final_data), allow_pickle = True)

    s4 = "\n\nEnd of the scattering search.\nTotal Time used = %.2fs"%(final_time - initial_time)
    s5 = "\nNumber of final events: %d/%d\n---------------------"%(len(result), len(number_events))
    print(s4)
    ef = 100*len(result)/len(number_events)
    print(s5)
    s6 = "Eficiência = %.2f%%"%ef
    print(s6)
    s7 = "Minutes: %.2f"%((final_time - initial_time)/60)
    print(s7)
    print("=========================================")


    log = s1 + s2 + s3 + s4 + s5 + s6 + "\n" + s7
    if logger == True:
        now = r"/home/ferrari/Mestrado/Track/logs/" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S%f") + ".txt"
        f = open(now, "w")
        f.write(log)
        f.close()


    if verb == True:
        pass
        # for data in final_data:
        #     clusters = np.copy(data[0])
        #     coefs = np.copy(data[1])
        #     plot_4DLine(clusters, coefs)

    # set_numbers = set(number_event)
    number_arr  = np.array(number_event)
    if comp == True:
        for ind, ii in zip(number_events, range(len(number_event))):
            if (ind in number_event) == False:
                compare(raw_data = events[ii], cluster = [], title = " Não passou. %s"%ind)
            else:
                j = np.where(number_arr == ind)[0][0]
                compare(raw_data = events[ii], cluster = result[j], title = "%s"%ind)
        # for ind, data in zip(number_event, result):
        #     compare(raw_data = events[ind], cluster = data, title = "%s"%ind)
    
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
    # np.savez_compressed("hist2d_charges_vs_angles.npz", total_charges = total_charges,
    #  total_angles = total_angles, total_vertex = total_vertex, total_length = total_length,
    #  total_points = total_points, total_info = total_info)
    # fig = plt.figure(dpi = 180)
    # hist, xbins, ybins, im = plt.hist2d(total_charges, total_angles, norm=mpl.colors.LogNorm(), bins = [50, 50], cmin = 1, cmap = "Accent")
    # fig.show()

def get_inliers(clusters, ve_filter = 110., samples = 26, min_neigh = 4, search_radius = 30, It = 800, dist = 15., mode = 3, th = 26.25, inner = 9., diam = 30.):
    result  = []
    infos   = []
    line_algo  = Line7(It, dist, samples, mode)
    t0 = time()
    tam_bar = 30
    MAX_TAM = len(clusters)
    for i, dados in enumerate(clusters):
        update_bar(i*100/MAX_TAM, tam_bar, end = "\r")
        data = dados[:, :4]
        # [0] = Amplitude_pulso, [1] = Raio, [2] = Numero do evento, [3] = Indice nuvem original
        # extra_info = np.hstack((dados[:, 4:], np.arange(len(dados)).reshape(-1, 1))) 
        index = np.where(data[:, 3] >= ve_filter)[0]
        if len(index) >= samples:
            aux = data[index]
            # extra_info = extra_info[index]
            new_cluster = o3d_outlier_removal(aux, min_points = min_neigh, search_radius = search_radius)
            if len(new_cluster) >= samples:
                # cluster = get_4inliersPY(new_cluster, line_algo, min_samples=samples)
                cluster = line_algo.fit(new_cluster)
                # for track in cluster:
                #     print(track.TCharge)
                #     print(return_inliers(cluster, data))
                # compare_ATTPC(new_cluster, cluster)
                if len(cluster) > 1:
                    cluster2 = clustering_algo(deepcopy(cluster), th, inner)
                    if 1 < len(cluster2):
                        cluster3 = redefine_vertex_tracks(deepcopy(cluster2))
                        cluster4 = select_tracks(cluster3, diametro = diam)
                        if len(cluster4) > 1:
                            for track in cluster4:
                                track.calculate_length()
                                # if track.length < 300 and (track.TCharge/len(track.data)) > 2000:
                                #     print(f"Carga total = {track.TCharge}. Número total de pontos da track = {len(track.data)}. Razão = {track.TCharge/len(track.data)}")
                                #     print(f"Cargas\n")
                                #     print(f"{track.data}")
                                #     print(f"{track.data[:, 4].max()}")
                                    # compare_ATTPC(data, cluster4)
                            # inliers.append(return_inliers(cluster4, data))
                            inliers = return_inliers(cluster4, data)
                            infos.append(get_info(dados, inliers, cluster4))
                            result.append(cluster4)
                        elif len(cluster4) == 1 and cluster4[0].isprimary == False:
                            cluster4[0].calculate_length()
                            # if cluster4[0].length < 300 and (cluster4[0].TCharge/len(cluster4[0].data)) >2000:
                            #     print(f"Carga total = {cluster4[0].TCharge}. Número total de pontos da track = {len(cluster4[0].data)}. Razão = {cluster4[0].TCharge/len(cluster4[0].data)}")
                            #     compare_ATTPC(data, cluster4)
                            # inliers.append(return_inliers(cluster4, data))
                            inliers = return_inliers(cluster4, data)
                            infos.append(get_info(dados, inliers, cluster4))
                            result.append(cluster4)
                        else:
                            infos.append(get_info(dados, np.array([], dtype = int)))
                    elif len(cluster2) == 1 and cluster2[0].isprimary == False:
                        cluster2[0].calculate_length()
                        # if cluster2[0].length < 300 and (cluster2[0].TCharge/len(cluster2[0].data)) >2000:
                        #     print(f"Carga total = {cluster2[0].TCharge}. Número total de pontos da track = {len(cluster2[0].data)}. Razão = {cluster2[0].TCharge/len(cluster2[0].data)}")
                        #     compare_ATTPC(data, cluster2)
                        # inliers.append(return_inliers(cluster2, data))
                        inliers = return_inliers(cluster2, data)
                        infos.append(get_info(dados, inliers, cluster2))
                        result.append(cluster2)
                    else:
                        infos.append(get_info(dados, np.array([], dtype = int)))
                elif len(cluster) == 1:
                    if cluster[0].isprimary == False and cluster[0].vinchamber == True and cluster[0].get_min_dist() <= diam:
                        cluster[0].calculate_length()
                        # if cluster[0].length < 300 and (cluster[0].TCharge/len(cluster[0].data)) >2000:
                        #     print(f"Carga total = {cluster[0].TCharge}. Número total de pontos da track = {len(cluster[0].data)}. Razão = {cluster[0].TCharge/len(cluster[0].data)}")
                        #     compare_ATTPC(data, cluster)
                        # inliers.append(return_inliers(cluster, data))
                        inliers = return_inliers(cluster, data)
                        infos.append(get_info(dados, inliers, cluster))
                        result.append(cluster)
                    else:
                        infos.append(get_info(dados, np.array([], dtype = int)))
                else:
                    infos.append(get_info(dados, np.array([], dtype = int)))
            else:
                infos.append(get_info(dados, np.array([], dtype = int)))
        else:
            infos.append(get_info(dados, np.array([], dtype = int)))
    # np.save("inliers_filtro.npy", np.array(inliers, dtype = object))
    update_bar(100, tam_bar, end = "\n")
    print(f"Eficiência = {len(result)*100/len(clusters)}%")
    t_f = time() - t0
    print(f"Tempo = {t_f} s = {t_f/60} min")
    print(f"Tamanho original = {len(cluster)}. Tamanho infos = {len(infos)}")
    return np.array(infos, dtype = object)

def update_bar(percent, tam_bar, end = "\r"):
    num_chars = int(percent*tam_bar/100)
    print("[" + "#"*num_chars + " "*(tam_bar - num_chars) + f"] {percent:.2f}%", end = end)

def return_inliers(cluster, data):
    inliers = []
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    q = data[:, 3]
    for track in cluster:
        for i in range(len(track.data)):
            inliers.append(np.where((x == track.data[i, 0]) & (y == track.data[i, 1]) & (z == track.data[i, 2]) & (q == track.data[i, 3]))[0][0])
    return np.array(inliers, dtype = int)

def scattering_filter2(events, run, ve_filter = 100., search_radius = 12., min_neigh = 4, samples = 30, dist = 10, th = 15., inner = 15., verb = False, part = 0, It = 1000, **kwargs):
    """
    Receive the data and select only the ones with scattering. Divided in three steps:
    First:  Select points with only a minimum energy.
    Second: Use outlier removal to clean more outliers.
    Third:  Applies ransac with clustering to get the trajectories and select cluster with two or more tracks.

    Parameters
    ----------

    events : list
        Set of all data to be filtered.
    run : int
        Number of the run.
    ve_filter : int or float, optional
        Energy filter. The default is 100.
    search_radius : int or float, optional
        Search radius for outlier removal. The default is 12.
    min_neigh : int, optional
        Minimum of neighbors for outlier removal. The default is 4.
    samples : int, optional
        Parameter of ransac. The default is 30.
    dist : int, optional
        Parameter of ransac. The default is 10.
    th : float, optional
        Threshold, parameter for clustering after ransac. The default is 10.
    inner : float, optional
        Parameter for clustering after ransac. The default is 15.
    verb : bool, optional
        If True plto all the scattering events. The default is False.
    part : int, optional
        Parameter to save files. The default is 0.
    It : int, optional
        Number of ransac iterations. The default is 1000.

    Returns
    -------
    None.

    """
    save   = kwargs.pop('save', False)
    comp   = kwargs.pop('comp', False)
    mode   = kwargs.pop('mode', 1)
    logger = kwargs.pop('comp', True)
    diam   = kwargs.pop('diametro', 30.)

    line_algo = Line6(It, dist, samples, inner, th, mode)

    s1 = "\n\tStarting the Scattering search.\n\n"
    
    print(s1)
    
    param = "\nve_filter = %f | search_radius = %f | min_neigh = %d | samples = %d | "%(ve_filter, search_radius, min_neigh, samples)
    param += "dist = %d | th = %f | inner = %f | It = %d"%(dist, th, inner, It)
    number_events = range(len(events))
    
    s2 = "\nInformations:\nRun number %d\nNumber of events: %d\n\n"%(run, len(number_events))
    print(s2)
    
    # Apllying the energy filter

    initial_time = time()
    s3 = "Starting the search."
    print(s3)
    result       = list()
    number_event = list()
    for data, number in zip(events, number_events):
        aux = list()
        # Energy filter
        for i in range(len(data)):
            if data[i][3] >= ve_filter:
                aux.append(data[i])
        if len(aux) >= samples:
            # Outlier removal
            aux = np.array(aux, dtype = float, ndmin = 2)
            # Abaixo opção de plot vs
            # plot3dvs(data, aux)
            new_cluster = o3d_outlier_removal(aux, min_points = min_neigh, search_radius = search_radius)
            # Abaixo opção de plot vs
            # plot3dvs(data, new_cluster)
            # print("Tamanho do cluster de número %d: %d"%(number, len(new_cluster)))
            if len(new_cluster) >= samples:
                # Ransac + clustering
                cluster = get_4inliersPY(new_cluster, line_algo, min_samples=samples)
                # compare(raw_data = new_cluster, cluster = cluster, title = "%s"%number)
                if 1 < len(cluster) < 6:
                    cluster2 = redefine_vertex_tracks(deepcopy(cluster))
                    # compare_clustering(cluster2, cluster3, number = str(number))
                    # result.append(cluster3)
                    # # result.append(cluster2)
                    # number_event.append(number)
                    cluster3 = select_tracks(cluster2, diametro = diam)
                    if len(cluster3) > 1:
                        # compare_clustering(cluster3, cluster4, number = str(number))
                        result.append(cluster3)
                        number_event.append(number)
                    elif len(cluster3) == 1 and cluster3[0].isprimary == False:
                        # compare_clustering(cluster3, cluster4, number = str(number))
                        result.append(cluster3)
                        number_event.append(number)
                elif len(cluster) == 1:
                    if cluster[0].isprimary == False and cluster[0].vinchamber == True and cluster[0].get_min_dist() <= diam:
                        result.append(cluster)
                        number_event.append(number)


    final_time = time()

    
    if save == True:
        final_data = list()
        for cluster in result:
            aux = list()
            for track in cluster:
                aux.append(track.get_data())
                aux.append(track.get_versor())
                aux.append(track.get_pb())
            final_data.append(np.array(aux))
        np.save("scattering_events_%d_%d"%(run, part), np.array(final_data), allow_pickle = True)

    s4 = "\n\nEnd of the scattering search.\nTotal Time used = %.2fs"%(final_time - initial_time)
    s5 = "\nNumber of final events: %d/%d\n---------------------"%(len(result), len(number_events))
    print(s4)
    print(s5)
    s6 = "Minutes: %.2f"%((final_time - initial_time)/60)
    print(s6)
    

    log = s1 + s2 + s3 + s4 + s5 + s6
    if logger == True:
        now = r"/home/ferrari/Mestrado/Track/logs/" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S%f") + ".txt"
        f = open(now, "w")
        f.write(log)
        f.close()

    
    if verb == True:
        for data in final_data:
            clusters = np.copy(data[0])
            coefs = np.copy(data[1])
            plot_4DLine(clusters, coefs)
    
    set_numbers = set(number_event)
    number_arr  = np.array(number_event)
    if comp == True:
        for ind in range(len(events)):
            if (ind in number_event) == False:
                compare(raw_data = events[ind], cluster = [], title = " Não passou. %s"%ind)
            else:
                j = np.where(number_arr == ind)[0][0]
                compare(raw_data = events[ind], cluster = result[j], title = "%s"%ind)

    #return final_data

def make_data(events, number_events, run, ve_filter = 100., search_radius = 12., min_neigh = 4, samples = 30, **kwargs):

    branches   = kwargs.pop('branches', None)
    all_events = kwargs.pop("all_events", [])
    # line_algo  = Line7(It, dist, samples, mode)

    s1 = "\n\tStarting the Scattering search.\n\n"
    
    print(s1)
    
    param = "\nve_filter = %f | search_radius = %f | min_neigh = %d | samples = %d | "%(ve_filter, search_radius, min_neigh, samples)
    # param += "dist = %d | th = %f | inner = %f | It = %d"%(dist, th, inner, It)
    # number_events = range(len(events))
    
    s2 = "\nInformations:\nRun number %d\nNumber of events: %d\n\n"%(run, len(number_events))
    print(s2)
    
    # Apllying the energy filter

    initial_time = time()
    s3 = "Starting the search."

    root_file = ROOT.TFile("/home/ferrari/Mestrado/Track/RF_TRAIN_DATA_17F_%d.root"%run,
                            "RECREATE")
    tree      = ROOT.TTree("tree", "%d"%run)


    num_inliers  = ROOT.std.vector("int")()
    vpx          = ROOT.std.vector("double")()
    vpy          = ROOT.std.vector("double")()
    vpz          = ROOT.std.vector("double")()
    vpt          = ROOT.std.vector("double")()
    vpe          = ROOT.std.vector("double")()
    inliers_R    = ROOT.std.vector("int")()

    tree.Branch("Num_inliers", num_inliers)
    tree.Branch("vpx", vpx)
    tree.Branch("vpy", vpy)
    tree.Branch("vpz", vpz)
    tree.Branch("vpt", vpt)
    tree.Branch("vpe", vpe)
    tree.Branch("inliers", inliers_R)

    print(s3)
    result       = list()
    number_event = list()
    inliers      = list()
    final_data   = list()
    for data, number in zip(events, number_events):
        # print("Análise evento ", number)
        aux = list()
        # Energy filter
        for i in range(len(data)):
            if data[i, 3] >= ve_filter:
                aux.append(data[i])
        # print("Filtro de energia ok")
        if len(aux) >= samples:#:
            # Outlier removal
            aux = np.array(aux, dtype = float, ndmin = 2)
            # Abaixo opção de plot vs
            # plot3dvs(data, aux)
            new_cluster     = o3d_outlier_removal(aux, min_points = min_neigh, search_radius = search_radius)
            # parcial_inliers = list()
            if len(new_cluster > 0):
                parcial_inliers = [np.where((data[:,0]==new_cluster[i, 0]) & (data[:,1]==new_cluster[i, 1]) & (data[:,2]==new_cluster[i, 2]) & (data[:,3]==new_cluster[i, 3]))[0][0] for i in range(len(new_cluster))]
                inliers.append(parcial_inliers)
                final_data.append(data)
                for ii in range(len(parcial_inliers)):
                    inliers_R.push_back(ROOT.int(parcial_inliers[ii]))
                num_inliers.push_back(ROOT.int(len(parcial_inliers)))
                for kk in range(len(branches["vpx"][number])):
                    try:
                        vpx.push_back(ROOT.double(branches["vpx"][number][kk]))
                        vpy.push_back(ROOT.double(branches["vpy"][number][kk]))
                        vpz.push_back(ROOT.double(branches["vpz"][number][kk]))
                        vpt.push_back(ROOT.double(branches["vpt"][number][kk]))
                        vpe.push_back(ROOT.double(branches["vpe"][number][kk]))
                    except:
                        pass
                tree.Fill()
                vpx.clear()
                vpy.clear()
                vpz.clear()
                vpt.clear()
                vpe.clear()
                inliers_R.clear()
                num_inliers.clear()
    tree.Write()    
                # parcial_inliers.append(aa)
            # print(parcial_inliers)
            # print(len(new_cluster), len(parcial_inliers))
            # print("Outlier removal ok")
            # Abaixo opção de plot vs
            # plot3dvs(data, new_cluster)
            # print("Tamanho do cluster de número %d: %d"%(number, len(new_cluster)))

    final_time = time()

    final_data = np.array(final_data, dtype = object)
    inliers    = np.array(inliers, dtype = object)
    np.save("/home/ferrari/Mestrado/Track/RF_INPUT_DATA_17F_%d.npy"%run, final_data, allow_pickle=True)
    np.save("/home/ferrari/Mestrado/Track/RF_TARGET_DATA_17F_%d.npy"%run, inliers, allow_pickle=True)

    s4 = "\n\nEnd of the scattering search.\nTotal Time used = %.2fs"%(final_time - initial_time)
    # s5 = "\nNumber of final events: %d/%d\n---------------------"%(len(result), len(number_events))
    print(s4)
    # ef = 100*len(result)/len(number_events)
    # print(s5)
    # s6 = "Eficiência = %.2f%%"%ef
    # print(s6)
    s7 = "Minutes: %.2f"%((final_time - initial_time)/60)
    print(s7)
    print("=========================================")


    # log = s1 + s2 + s3 + s4 + s5 + s6 + "\n" + s7
    # if logger == True:
    #     now = r"/home/ferrari/Mestrado/Track/logs/" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S%f") + ".txt"
    #     f = open(now, "w")
    #     f.write(log)
    #     f.close()

def get_number_tracks(events, number_events, run, ve_filter = 100., search_radius = 12., min_neigh = 4, samples = 30, dist = 10, th = 15., inner = 15., verb = False, part = 0, It = 1000, **kwargs):
    """
    Receive the data and select only the ones with scattering. Divided in three steps:
    First:  Select points with only a minimum energy.
    Second: Use outlier removal to clean more outliers.
    Third:  Applies ransac with clustering to get the trajectories and select cluster with two or more tracks.

    Parameters
    ----------

    events : list
        Set of all data to be filtered.
    run : int
        Number of the run.
    ve_filter : int or float, optional
        Energy filter. The default is 100.
    search_radius : int or float, optional
        Search radius for outlier removal. The default is 12.
    min_neigh : int, optional
        Minimum of neighbors for outlier removal. The default is 4.
    samples : int, optional
        Parameter of ransac. The default is 30.
    dist : int, optional
        Parameter of ransac. The default is 10.
    th : float, optional
        Threshold, parameter for clustering after ransac. The default is 10.
    inner : float, optional
        Parameter for clustering after ransac. The default is 15.
    verb : bool, optional
        If True plto all the scattering events. The default is False.
    part : int, optional
        Parameter to save files. The default is 0.
    It : int, optional
        Number of ransac iterations. The default is 1000.

    Returns
    -------
    None.

    """
    mode       = kwargs.pop('mode', 1)
    line_algo  = Line7(It, dist, samples, mode)

    s1 = "\n\tStarting the Scattering search.\n\n"
    
    print(s1)
    
    param = "\nve_filter = %f | search_radius = %f | min_neigh = %d | samples = %d | "%(ve_filter, search_radius, min_neigh, samples)
    param += "dist = %d | th = %f | inner = %f | It = %d"%(dist, th, inner, It)
    # number_events = range(len(events))
    
    s2 = "\nInformations:\nRun number %d\nNumber of events: %d\n\n"%(run, len(number_events))
    print(s2)
    s3 = "Starting the search."
    print(s3)
    result       = list()
    number_event = list()
    for data, number in zip(events, number_events):
        # print("Análise evento ", number)
        # Plot_Cluster_raw(data, number = number)
        # Energy filter
        index = np.where(data[:, 3] >= ve_filter)[0]
        # print("Filtro de energia ok")
        if len(index) >= samples:#:
            # Outlier removal
            aux = data[index]
            # Abaixo opção de plot vs
            # plot3dvs(data, aux)
            new_cluster = o3d_outlier_removal(aux, min_points = min_neigh, search_radius = search_radius)
            # Abaixo opção de plot vs
            # plot3dvs(data, new_cluster)
            if len(new_cluster) >= samples:
                # Ransac + clustering
                cluster = get_4inliersPY(new_cluster, line_algo, min_samples=samples)
                # compare(new_cluster, cluster, title = str(number))
                if len(cluster) > 1:
                    cluster2 = clustering_algo(deepcopy(cluster), th, inner)
                    result.append(len(cluster2))
                    number_event.append(number)
                elif len(cluster) == 1:
                    # if cluster[0].isprimary == False and cluster[0].vinchamber == True and cluster[0].get_min_dist() <= diam:
                    result.append(1)
                    number_event.append(number)
            else:
                result.append(0)
                number_event.append(number)
    result       = np.array(result, dtype = int)
    number_event = np.array(number_event, dtype = int)
    np.savez_compressed(f"event_{run}_number_of_tracks.npz", result)
    np.savez_compressed(f"event_{run}_event_number.npz", number_event)

def scattering_filter_clustering(events, number_events, run, ve_filter = 100., search_radius = 12., min_neigh = 4, samples = 30, dist = 10, th = 15., inner = 15., verb = False, part = 0, It = 1000, **kwargs):
    """
    Receive the data and select only the ones with scattering. Divided in three steps:
    First:  Select points with only a minimum energy.
    Second: Use outlier removal to clean more outliers.
    Third:  Applies ransac with clustering to get the trajectories and select cluster with two or more tracks.

    Parameters
    ----------

    events : list
        Set of all data to be filtered.
    run : int
        Number of the run.
    ve_filter : int or float, optional
        Energy filter. The default is 100.
    search_radius : int or float, optional
        Search radius for outlier removal. The default is 12.
    min_neigh : int, optional
        Minimum of neighbors for outlier removal. The default is 4.
    samples : int, optional
        Parameter of ransac. The default is 30.
    dist : int, optional
        Parameter of ransac. The default is 10.
    th : float, optional
        Threshold, parameter for clustering after ransac. The default is 10.
    inner : float, optional
        Parameter for clustering after ransac. The default is 15.
    verb : bool, optional
        If True plto all the scattering events. The default is False.
    part : int, optional
        Parameter to save files. The default is 0.
    It : int, optional
        Number of ransac iterations. The default is 1000.

    Returns
    -------
    None.

    """
    comp       = kwargs.pop('comp', False)
    mode       = kwargs.pop('mode', 1)
    logger     = kwargs.pop('logger', True)
    diam       = kwargs.pop('diametro', 30.)
    line_algo  = Line7(It, dist, samples, mode)

    s1 = "\n\tStarting the Scattering search.\n\n"
    
    print(s1)
    
    param = "\nve_filter = %f | search_radius = %f | min_neigh = %d | samples = %d | "%(ve_filter, search_radius, min_neigh, samples)
    param += "dist = %d | th = %f | inner = %f | It = %d"%(dist, th, inner, It)
    # number_events = range(len(events))
    
    s2 = "\nInformations:\nRun number %d\nNumber of events: %d\n\n"%(run, len(number_events))
    print(s2)
    
    # Apllying the energy filter

    initial_time = time()
    s3 = "Starting the search."
    print(s3)
    result       = list()
    number_event = list()
    for data, number in zip(events, number_events):
        # print("Análise evento ", number)
        # Plot_Cluster_raw(data, number = number)
        # Energy filter
        index = np.where(data[:, 3] >= ve_filter)[0]
        # print("Filtro de energia ok")
        if len(index) >= samples:#:
            # Outlier removal
            aux = data[index]
            # Abaixo opção de plot vs
            # plot3dvs(data, aux)
            new_cluster = o3d_outlier_removal(aux, min_points = min_neigh, search_radius = search_radius)
            # print("Outlier removal ok")
            # Abaixo opção de plot vs
            # plot3dvs(data, new_cluster)
            # print("Tamanho do cluster de número %d: %d"%(number, len(new_cluster)))
            if len(new_cluster) >= samples:
                # Ransac + clustering
                cluster = get_4inliersPY(new_cluster, line_algo, min_samples=samples)
                # Plot_Cluster(cluster)
                # compare(new_cluster, cluster, title = str(number))
                # print("Ransac ok")
                if len(cluster) > 1:
                    cluster2 = clustering_algo(deepcopy(cluster), th, inner)
                    Plot_Cluster(cluster2)
                    # compare_clustering(cluster, cluster2, number = str(number))
                    # if len(new_cluster) < samples:
                    #     compare(data, cluster2, title = str(number) + "Não deu certo")
                    # else:
                    #     compare(data, cluster2, title = str(number))
                    if 1 < len(cluster2):
                        cluster3 = redefine_vertex_tracks(deepcopy(cluster2))
                        # compare_clustering(cluster2, cluster3, number = str(number))
                        # result.append(cluster3)
                        # # result.append(cluster2)
                        # number_event.append(number)
                        cluster4 = select_tracks(cluster3, diametro = diam)
                        # compare_clustering(cluster3, cluster4, number = str(number))
                        if len(cluster4) > 1:
                            # compare_clustering(cluster3, cluster4, number = str(number))
                            # for reta in cluster4:
                            #     reta.calculate_length()
                            #     reta.set_inliers(data)
                            # print("Evento %d aceito"%number)
                            result.append(cluster4)
                            number_event.append(number)
                        elif len(cluster4) == 1 and cluster4[0].isprimary == False:
                            # print("Evento %d aceito"%number)
                            cluster4[0].calculate_length()
                            # cluster[0].set_inliers(data)
                            # compare_clustering(cluster3, cluster4, number = str(number))
                            result.append(cluster4)
                            number_event.append(number)

                    elif len(cluster2) == 1:
                        if cluster2[0].isprimary == False and cluster[0].vinchamber == True and cluster[0].get_min_dist() <= diam:
                            # print("Evento %d aceito"%number)
                            cluster2[0].calculate_length()
                            # cluster[0].set_inliers(data)
                            result.append(cluster2)
                            number_event.append(number)
                elif len(cluster) == 1:
                    if cluster[0].isprimary == False and cluster[0].vinchamber == True and cluster[0].get_min_dist() <= diam:
                        # print("Evento %d aceito"%number)
                        cluster[0].calculate_length()
                        # cluster[0].set_inliers(data)
                        result.append(cluster)
                        number_event.append(number)
                # else:
                #     compare(data, cluster2, title = str(number) + "Não deu certo: " + str(len(data)))

    final_time = time()

    s4 = "\n\nEnd of the scattering search.\nTotal Time used = %.2fs"%(final_time - initial_time)
    s5 = "\nNumber of final events: %d/%d\n---------------------"%(len(result), len(number_events))
    print(s4)
    ef = 100*len(result)/len(number_events)
    print(s5)
    s6 = "Eficiência = %.2f%%"%ef
    print(s6)
    s7 = "Minutes: %.2f"%((final_time - initial_time)/60)
    print(s7)
    print("=========================================")


    log = s1 + s2 + s3 + s4 + s5 + s6 + "\n" + s7
    if logger == True:
        now = r"/home/ferrari/Mestrado/Track/logs/" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S%f") + ".txt"
        f = open(now, "w")
        f.write(log)
        f.close()


    if verb == True:
        pass
        # for data in final_data:
        #     clusters = np.copy(data[0])
        #     coefs = np.copy(data[1])
        #     plot_4DLine(clusters, coefs)

    number_arr  = np.array(number_event)
    if comp == True:
        for ind, ii in zip(number_events, range(len(number_event))):
            if (ind in number_event) == False:
                compare(raw_data = events[ii], cluster = [], title = " Não passou. %s"%ind)
            else:
                j = np.where(number_arr == ind)[0][0]
                compare(raw_data = events[ii], cluster = result[j], title = "%s"%ind)
        # for ind, data in zip(number_event, result):
        #     compare(raw_data = events[ind], cluster = data, title = "%s"%ind)