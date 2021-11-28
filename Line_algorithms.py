#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:58:45 2021

@author: Guilherme Ferrari Fortino

Adaptation from https://github.com/jczamorac/Tracking_RANSAC

"""

import numpy as np
import ctypes
import matplotlib.pyplot as plt
import tpc_utils as utils
from track import Track
# from operator import itemgetter

class Line():
    """
    Ransac algorithm to fit a line with options to random sampling and
    criteria for the best model.
    mode = 0 -> Chooses two random points from cluster.
    mode = 1 -> Chooses two random points based on gaussian sampling.
    mode = 2 -> Chooses two random points based on weight/charge sampling.
    mode = 3 -> Chooses two random points based on weight and gaussian sampling.
    
    Parameters
    ----------
    
    data : numpy array
        Array with the cluster.
    number_it : int
        Number of iterations.
    min_dist : flaot
        Minimum distance from point to line.
    charge : numpy array
        Charge / weights from the dataset.
    mode : int, optional
        Random sampling mode. The default is 0.
    selection_mode : int, optional
        Parameter of ransac best model. If 0 the best model is the one with
        the most inliers. If 1 the best model is the one with the minor sum of
        the squared distances / number of inliers. The default is 0.
    
    Returns
    -------
    inliers: numpy array
        Inliers coefficients.
    versor: numpy array
        Versor of the best model.
    point: numpy array
        Reference point where it intercepts the line.
    """
    def __init__(self, data, number_it, min_dist, charge, mode: int = 0, selection_mode: int = 0):
        #print("\n\n", len(data), "\n")
        self.data      = data.copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.number_it = ctypes.c_int(number_it)
        self.min_dist  = ctypes.c_double(min_dist)
        self.charge    = charge.copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.versor    = np.zeros(3).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.inliers   = np.zeros(len(data)).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.pb        = np.zeros(3).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.size      = ctypes.c_int(len(data))
        self.mode      = ctypes.c_int(mode)
        self.sm        = ctypes.c_int(selection_mode)
        arq            = ctypes.CDLL("./line.so")
        self.model     = None
        if selection_mode == 0:
            self.model = arq.Ransac
        else:
            self.model = arq.Ransac_2
        self.model.argtypes = [ctypes.POINTER(ctypes.c_double), # double (*data)[3]
                               ctypes.POINTER(ctypes.c_double), # double *versor
                               ctypes.POINTER(ctypes.c_double), # double* pb
                               ctypes.POINTER(ctypes.c_int),    # int *inliers
                               ctypes.POINTER(ctypes.c_double), # charge (weights)
                               ctypes.c_int,                    # int number_it
                               ctypes.c_double,                 # double min_dist
                               ctypes.c_int,                    # int size
                               ctypes.c_int]                    # int mode

        self.restype = ctypes.c_int
        
    def fit(self):
        num_inliers = int(self.model(self.data, self.versor, self.pb, self.inliers,
                                     self.charge, self.number_it, self.min_dist,
                                     self.size, self.mode))
        inliers = np.array([int(self.inliers[i]) for i in range(num_inliers)])
        versor  = np.array([float(self.versor[0]), float(self.versor[1]), float(self.versor[2])])
        pb      = np.array([float(self.pb[0]), float(self.pb[1]), float(self.pb[2])])
        return inliers, versor, pb

class Line2():
    """
    Ransac algorithm to fit a line with options to random sampling and
    criteria for the best model.
    mode = 0 -> Chooses two random points from cluster.
    mode = 1 -> Chooses two random points based on gaussian sampling.
    mode = 2 -> Chooses two random points based on weight/charge sampling.
    mode = 3 -> Chooses two random points based on weight and gaussian sampling.
    
    Parameters
    ----------
    
    data : numpy array
        Array with the cluster.
    number_it : int
        Number of iterations.
    min_dist : flaot
        Minimum distance from point to line.
    charge : numpy array
        Charge / weights from the dataset.
    mode : int, optional
        Random sampling mode. The default is 0.
    selection_mode : int, optional
        Parameter of ransac best model. If 0 the best model is the one with
        the most inliers. If 1 the best model is the one with the minor sum of
        the squared distances / number of inliers. The default is 0.
    
    Returns
    -------
    inliers: numpy array
        Inliers coefficients.
    versor: numpy array
        Versor of the best model.
    point: numpy array
        Reference point where it intercepts the line.
    """
    def __init__(self, data, number_it, min_dist, charge, mode: int = 0):
        #print("\n\n", len(data), "\n")
        #self.data      = data.tolist()
        self.number_it = number_it
        self.min_dist  = min_dist
        #self.charge    = charge.tolist()
        self.versor    = [0]*3
        #self.inliers   = [0]*len(data)
        self.pb        = [0]*3
        #self.size      = ctypes.c_int(len(data))
        self.mode      = ctypes.c_int(mode)
        #self.sm        = ctypes.c_int(selection_mode)
        arq            = ctypes.PyDLL("/home/ferrari/Mestrado/Track/py_cpp/line_py.so")
        self.model     = arq.Ransac_3
        self.model.argtypes = [ctypes.py_object,    # data
                               ctypes.py_object,    # versor
                               ctypes.py_object,    # pb
                               ctypes.py_object,    # inliers
                               ctypes.py_object,    # charge (weights)
                               ctypes.py_object,    # number_it
                               ctypes.py_object,    # min_dist
                               ctypes.c_int]        # int mode

        self.restype = ctypes.c_int
        
    def fit(self):
        num_inliers = int(self.model(self.data, self.versor, self.pb, self.inliers,
                                     self.charge, self.number_it, self.min_dist,
                                     self.mode))
        inliers = np.array(self.inliers[:int(num_inliers)], dtype = int)
        versor  = np.array(self.versor)
        pb      = np.array(self.pb)
        return inliers, versor, pb
        
class Line3():
    """
    Ransac algorithm to fit a line with options to random sampling and
    criteria for the best model.
    mode = 0 -> Chooses two random points from cluster.
    mode = 1 -> Chooses two random points based on gaussian sampling.
    mode = 2 -> Chooses two random points based on weight/charge sampling.
    mode = 3 -> Chooses two random points based on weight and gaussian sampling.
    
    Parameters
    ----------
    
    data : numpy array
        Array with the cluster.
    number_it : int
        Number of iterations.
    min_dist : flaot
        Minimum distance from point to line.
    charge : numpy array
        Charge / weights from the dataset.
    mode : int, optional
        Random sampling mode. The default is 0.
    selection_mode : int, optional
        Parameter of ransac best model. If 0 the best model is the one with
        the most inliers. If 1 the best model is the one with the minor sum of
        the squared distances / number of inliers. The default is 0.
    
    Returns
    -------
    inliers: numpy array
        Inliers coefficients.
    versor: numpy array
        Versor of the best model.
    point: numpy array
        Reference point where it intercepts the line.
    """
    def __init__(self, number_it, min_dist, mode: int = 0):
        #print("\n\n", len(data), "\n")
        # self.data      = data.copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.number_it = ctypes.c_int(number_it)
        self.min_dist  = ctypes.c_double(min_dist)
        # self.charge    = charge.copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.versor    = np.zeros(3).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #self.inliers   = np.zeros(len(data)).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.pb        = np.zeros(3).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # self.size      = ctypes.c_int(len(data))
        self.mode      = ctypes.c_int(mode)
        arq            = ctypes.CDLL("./line.so")
        self.model = arq.Ransac_2
        self.model.argtypes = [ctypes.POINTER(ctypes.c_double), # double (*data)[3]
                               ctypes.POINTER(ctypes.c_double), # double *versor
                               ctypes.POINTER(ctypes.c_double), # double* pb
                               ctypes.POINTER(ctypes.c_int),    # int *inliers
                               ctypes.POINTER(ctypes.c_double), # charge (weights)
                               ctypes.c_int,                    # int number_it
                               ctypes.c_double,                 # double min_dist
                               ctypes.c_int,                    # int size
                               ctypes.c_int]                    # int mode

        self.restype = ctypes.c_int
        
    def fit(self, data, charge):
        inliers = np.zeros(len(data)).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        num_inliers = int(self.model(data.copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     self.versor, self.pb, inliers,
                                     charge.copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                      self.number_it, self.min_dist,
                                     ctypes.c_int(len(data)), self.mode))
        inliers = np.array([int(inliers[i]) for i in range(num_inliers)])
        versor  = np.array([float(self.versor[0]), float(self.versor[1]), float(self.versor[2])])
        pb      = np.array([float(self.pb[0]), float(self.pb[1]), float(self.pb[2])])
        return inliers, versor, pb

class Line4():
    def __init__(self, number_it, min_dist, min_samples, mode: int = 0):
        self.number_it      = ctypes.c_int(number_it)
        self.min_d          = min_dist
        self.min_dist       = ctypes.c_double(min_dist)
        self.mode           = ctypes.c_int(mode)
        self.min_s          = min_samples
        self.min_samples    = ctypes.c_int(min_samples)
        arq                 = ctypes.PyDLL("./line.so")
        # arq                 = ctypes.CDLL("./line.so")
        self.model          = arq.Ransac_3
        self.model.argtypes = [ctypes.POINTER(ctypes.c_double), # double (*data)[3]
                               ctypes.py_object,                # list tudo
                               ctypes.POINTER(ctypes.c_double), # charge (weights)
                               ctypes.c_int,                    # int number_it
                               ctypes.c_double,                 # double min_dist
                               ctypes.c_int,                    # int size
                               ctypes.c_int,                    # int mode
                               ctypes.c_int]                    # int min_samples
        self.get_models          = arq.Get_Models
        self.get_models.argtypes = [ctypes.POINTER(ctypes.c_double), # double (*data)[4]
                                    ctypes.POINTER(ctypes.c_double), # double (*versores)[3]
                                    ctypes.POINTER(ctypes.c_double), # double (*points)[3] 
                                    ctypes.py_object,                # PyObject* inliers
                                    ctypes.py_object,                # PyObject* best_vp
                                    ctypes.c_int,                    # int size
                                    ctypes.c_int,                    # int num_models
                                    ctypes.c_double,                 # double min_dist
                                    ctypes.c_int]                    # int min_size
        
    def fit(self, data):
        tudo     = list()
        size     = ctypes.c_int(len(data))
        data_cpp = data.copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.model(data[:, :3].copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   tudo, data[:, 3].copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   self.number_it, self.min_dist, size, self.mode, self.min_samples)
        if len(tudo) == 0:
            return []
        pcloud     = np.copy(data)
        cluster    = list()
        tudo       = np.array(tudo)
        arg        = np.argsort(tudo[:, 0])
        tudo       = tudo[arg]
        # for i in range(0, len(tudo)):
        # for i in range(0, 10):
        #     plot_4DLine(pcloud, tudo[i, 1:4], tudo[i, 4:])
        versores   = tudo[:, 1:4].copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        points     = tudo[:, 4:].copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        num_models = ctypes.c_int(len(tudo))
        inliers    = list()
        vp         = list()
        # print(len(tudo))
        self.get_models(data_cpp, versores, points, inliers, vp, size,
                        num_models, self.min_dist, self.min_samples)
        # for i in vp:
        #     plot_4DLine(pcloud, np.array(i[:3]), np.array(i[3:]))
        if len(inliers) > 0:
            vp = np.array(vp)
            for i in range(len(inliers)):
                new_track = Track(np.array(inliers[i].copy()), vp[i, 0:3], vp[i, 3:6])
                cluster.append(new_track)
        return cluster 

class Line5():
    def __init__(self, number_it, min_dist, min_samples, mode: int = 0):
        self.number_it      = ctypes.c_int(number_it)
        self.min_dist       = ctypes.c_double(min_dist)
        self.mode           = ctypes.c_int(mode)
        self.min_samples    = ctypes.c_int(min_samples)
        arq                 = ctypes.PyDLL("./line.so")
        # arq                 = ctypes.CDLL("./line.so")
        self.model          = arq.Ransac_4
        self.model.argtypes = [ctypes.POINTER(ctypes.c_double), # double (*data)[4]
                               ctypes.py_object,                # PyObject* inliers
                               ctypes.py_object,                # PyObject* best_vp
                               ctypes.POINTER(ctypes.c_double), # double* charge          
                               ctypes.c_int,                    # int number_it
                               ctypes.c_double,                 # double min_dist
                               ctypes.c_int,                    # int mode
                               ctypes.c_int,                    # int min_inlier
                               ctypes.c_int]                    # int size
        
    def fit(self, data):
        cluster  = list()
        inliers  = list()
        vp       = list()
        size     = ctypes.c_int(len(data))
        data_cpp = data.copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #           dados,  inliers, v e p, cargas
        self.model(data_cpp, inliers, vp, data[:, 3].copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   self.number_it, self.min_dist, self.mode, self.min_samples, size)
        #            num_it,         min_dist,        modo random, min_inliers 
        if len(inliers) == 0:
            return []
        # for i in vp:
        #     plot_4DLine(pcloud, np.array(i[:3]), np.array(i[3:]))
        else:
            vp = np.array(vp)
            for i in range(len(inliers)):
                new_track = Track(np.array(inliers[i].copy()), np.array(vp[i][:3]), np.array(vp[i][3:]))
                cluster.append(new_track)
        return cluster 

class Line6():
    def __init__(self, number_it, min_dist, min_samples, inner, th, mode: int = 0):
        self.number_it      = ctypes.c_int(number_it)
        self.min_dist       = ctypes.c_double(min_dist)
        self.mode           = ctypes.c_int(mode)
        self.min_samples    = ctypes.c_int(min_samples)
        self.th             = ctypes.c_double(th)
        self.inner          = ctypes.c_double(inner)
        arq                 = ctypes.PyDLL("./line.so")
        # arq                 = ctypes.CDLL("./line.so")
        self.model          = arq.Ransac_clustering
        self.model.argtypes = [ctypes.POINTER(ctypes.c_double), # double (*data)[4]
                               ctypes.py_object,                # PyObject* inliers
                               ctypes.py_object,                # PyObject* best_vp
                               ctypes.POINTER(ctypes.c_double), # double* charge          
                               ctypes.c_int,                    # int number_it
                               ctypes.c_double,                 # double min_dist
                               ctypes.c_int,                    # int mode
                               ctypes.c_int,                    # int min_inlier
                               ctypes.c_int,                    # int size
                               ctypes.c_double,                 # double inner
                               ctypes.c_double]                 # double th

        
    def fit(self, data):
        if len(data) < 2:
            return []
        cluster  = list()
        inliers  = list()
        vp       = list()
        size     = ctypes.c_int(len(data))
        data_cpp = data.copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #           dados,  inliers, v e p, cargas
        self.model(data_cpp, inliers, vp, data[:, 3].copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   self.number_it, self.min_dist, self.mode, self.min_samples, size, self.inner, self.th)
        #            num_it,         min_dist,        modo random, min_inliers 
        if len(inliers) == 0:
            return []
        # for i in vp:
        #     plot_4DLine(pcloud, np.array(i[:3]), np.array(i[3:]))
        else:
            # print(vp)
            vp = np.array(vp)
            for i in range(len(inliers)):
                new_track = Track(np.array(inliers[i].copy()), np.array(vp[i][:3]), np.array(vp[i][3:]))
                cluster.append(new_track)
        return cluster

class Line7():
    __slots__ = ('number_it', 'min_dist', 'min_samples', 'mode')
    def __init__(self, number_it, min_dist, min_samples, mode: int = 0):
        self.number_it      = number_it   # ctypes.c_int(number_it)
        self.min_dist       = min_dist    # ctypes.c_double(min_dist)
        self.mode           = mode        # ctypes.c_int(mode)
        self.min_samples    = min_samples # ctypes.c_int(min_samples)
        # arq                 = ctypes.PyDLL("./line.so")
        # arq                 = ctypes.CDLL("./line.so")
        # self.model          = arq.Ransac_4_RF
        # self.model.argtypes = [ctypes.POINTER(ctypes.c_double), # double (*data)[4]
        #                        ctypes.py_object,                # PyObject* inliers
        #                        ctypes.py_object,                # PyObject* best_vp
        #                        ctypes.POINTER(ctypes.c_double), # double* charge          
        #                        ctypes.c_int,                    # int number_it
        #                        ctypes.c_double,                 # double min_dist
        #                        ctypes.c_int,                    # int mode
        #                        ctypes.c_int,                    # int min_inlier
        #                        ctypes.c_int,                    # int size
        #                        ctypes.py_object]                # PyObject* target

    def fit(self, data):
        cluster  = list()
        # inliers  = list()
        # vp       = list()
        # target   = list()
        # size     = ctypes.c_int(len(data))
        # data_cpp = data.copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #           dados,  inliers, v e p, cargas
        # self.model(data_cpp, inliers, vp, data[:, 3].copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        #            self.number_it, self.min_dist, self.mode, self.min_samples, size, target)
        #            num_it,         min_dist,        modo random, min_inliers
        # print(len(data))
        # inliers, versors, points = myModule.ransac(data.tolist(), self.number_it, self.min_dist, self.min_samples, self.mode)
        inliers, versors, points = utils.ransac(data, self.number_it, self.min_dist, self.min_samples, self.mode)
        if len(inliers) == 0:
            return []
        # for i in vp:
        #     plot_4DLine(pcloud, np.array(i[:3]), np.array(i[3:]))
        else:
            for i in range(len(inliers)):
                new_track = Track(data[inliers[i].astype(int)], versors[i], points[i])
                cluster.append(new_track)
        return cluster

def get_versor_cpp(data):
    '''
    3D line fit algorithm.

    Parameters
    ----------
    data : array
        4-vector dataset.

    Returns
    -------
    versor : np.ndarray
        3-vector versor of the dataset.
    Pb : 
        Point in space which intercepts the line.

    '''
    # Fit3D = ctypes.CDLL("./line.so").Fit3D
    # Fit3D.argtypes = [ctypes.POINTER(ctypes.c_double),  # double *vX
    #                   ctypes.POINTER(ctypes.c_double),  # double *vY
    #                   ctypes.POINTER(ctypes.c_double),  # double *vZ
    #                   ctypes.POINTER(ctypes.c_double),  # double *vQ
    #                   ctypes.POINTER(ctypes.c_double),  # double *versor
    #                   ctypes.POINTER(ctypes.c_double),  # double *Pb
    #                   ctypes.c_int]                     # int size
    # p = np.zeros(3).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # v = np.zeros(3).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # Fit3D(data[:, 0].copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #       data[:, 1].copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #       data[:, 2].copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #       data[:, 3].copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #       v, p, ctypes.c_int(len(data)))
    # versor = np.array([float(v[0]), float(v[1]), float(v[2])], dtype = float)
    # Pb     = np.array([float(p[0]), float(p[1]), float(p[2])], dtype = float)
    # versor, Pb = myModule.fit3D(data.tolist())
    # return np.array(versor, dtype = float), np.array(Pb, dtype = float)
    versor, Pb = utils.fit3D(data)
    return versor, Pb

def get_info(dados:np.ndarray, inliers:np.ndarray, cluster:"list[Track]" = []) -> np.ndarray:
    new_data = np.hstack((dados, np.zeros(len(dados), dtype = int).reshape(-1, 1)))
    new_data[inliers, -1] = 1
    if len(cluster) > 0:
        dummy = np.zeros(len(dados), dtype = int)
        lens  = np.zeros(len(dados), dtype = float)
        for i, track in enumerate(cluster):
            pontos = return_inliers(track, new_data)
            lens[pontos]  = track.length
            dummy[pontos] = i + 1
        new_data = np.hstack((new_data, dummy.reshape(-1, 1), lens.reshape(-1, 1)))
    else:
        new_data = np.hstack((new_data,
         np.zeros(len(dados), dtype = int).reshape(-1, 1),
         np.zeros(len(dados), dtype = int).reshape(-1, 1)))
    return new_data

def return_inliers(track, data):
    inliers = []
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    q = data[:, 3]
    for i in range(len(track.data)):
        inliers.append(np.where((x == track.data[i, 0]) & (y == track.data[i, 1]) & (z == track.data[i, 2]) & (q == track.data[i, 3]))[0][0])
    return np.array(inliers, dtype = int)
           

        