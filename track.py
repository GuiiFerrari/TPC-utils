#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 22:24:27 2021

@author: Guilherme Ferrari Fortino

Track class

"""

import numpy as np
from numpy import degrees, arccos, array, cross
from numpy.linalg import norm
# import ctypes
import tpc_utils as utils
from scipy.spatial import distance
# from Line_algorithms import get_versor_cpp

class Track():
    # __slots__ = ('data', 'versor', 'pb', 'TCharge', 'vertex', 'min_dist', 'inliers', 'angle', 'isprimary', 'vinchamber', "length", "phi", "theta", "maxPos")
    __slots__ = ('data', 'versor', 'pb', 'TCharge', 'vertex', 'min_dist', 'angle', 'isprimary', 'vinchamber', "length", "maxPos", "inliers")
    def __init__(self, data, versor, pb):
        # print("\n")
        # print(data.shape, data[0], data[1], data[2])
        # print(versor)
        self.data       = np.copy(data)
        # versor, pb      = get_versor_cpp(data)
        # print(versor)
        self.versor     = np.copy(versor)
        self.pb         = np.copy(pb)
        self.TCharge    = np.sum(data[:, 3])
        self.vertex     = np.array([1000., 1000., -1000.], dtype=float)
        self.min_dist   = 0.
        self.angle      = 0.
        self.vinchamber = False
        self.isprimary  = False
        self.length     = 0.
        # self.phi        = 0.
        # self.theta      = 0.
        self.maxPos     = np.array([1000., 1000., -1000.], dtype=float)
        self.inliers    = []
        self._set_properties()
        # print(self.angle)
        
        
    def __add__(self, track):
        new_data  = np.vstack((self.get_data(), track.get_data()))
        new_versor, new_pb = get_versor_cpp(new_data)
        new_track = Track(new_data, new_versor, new_pb)
        return new_track
    
    def __radd__(self, track):
        return self + track
    
    def _set_properties(self):
        primary    = np.array([0., 0., -1.], dtype = float)
        versor     = self.versor
        # print(versor)
        # p1, p2     = self.pb, np.array([-3.35, -0.855, 0.0], dtype = float)
        p1, p2     = self.pb, np.array([0., 0., 0.0], dtype = float)
        Vc         = cross(versor, primary)
        self.angle = degrees(arccos(np.abs(np.inner(versor, primary))))
        # print(self.angle)
        if self.angle != 0:
            #print(Vc)
            Vc = Vc/norm(Vc)
            #print(Vc)
            self.min_dist = np.abs(np.dot(Vc, p1 - p2))
            RHS = p2 - p1
            LHS = array([versor, -primary, Vc]).T
            t1, t2, _   = np.linalg.solve(LHS, RHS)
            point_line1 = p1 + t1*versor
            point_line2 = p2 + t2*primary
            self.vertex = 0.5*(point_line1 + point_line2)
            #print("entrou")
            vertex = self.vertex
            if np.abs(vertex[0]) < 250 and np.abs(vertex[1]) < 250 and vertex[2] > -400 and vertex[2] < 1200.:
                self.vinchamber = True
            if self.angle < 5.:
                # nd  = np.dot(primary, versor) # calculo da interseção de reta em plano
                # w   = p1 - p2
                # si  = - np.dot(primary, w) / nd
                # ponto = w + si * versor # + p2 (depois tirar p2)
                try:
                    # t     = (512. - p1[2])/versor[2]
                    t     = (0. - p1[2])/versor[2]
                    ponto = p1 + t*versor
                    # print(ponto)
                    # if (ponto[0] + 3.35)**2 + (ponto[1] + 0.855)**2 < 225.0: # 17F
                    if (ponto[0])**2 + (ponto[1])**2 < 225.0: # 14O
                        self.isprimary = True
                except:
                    pass
                
        else:
            # ponto = cross(versor, p2 - p1)
            self.min_dist  = norm(cross(versor, p2 - p1))
            if self.min_dist <= 25:
                self.isprimary = True

    def redefine_vertex(self, track):
        primary = track.get_versor()
        versor     = self.versor
        p1, p2     = self.pb, track.get_pb()
        Vc         = cross(versor, primary)
        self.angle = degrees(arccos(np.abs(np.inner(versor, primary))))
        if self.angle == 0.:
            self.min_dist  = norm(cross(versor, p2 - p1))
        else:
            Vc = Vc/norm(Vc)
            self.min_dist = np.abs(np.dot(Vc, p1 - p2))
        RHS = p2 - p1
        LHS = array([versor, -primary, Vc]).T
        try:
            t1, t2, _   = np.linalg.solve(LHS, RHS)
            point_line1 = p1 + t1*versor
            point_line2 = p2 + t2*primary
            self.vertex = 0.5*(point_line1 + point_line2)
            # print(self.vertex)
            if np.abs(self.vertex[0]) < 140 and np.abs(self.vertex[1]) < 140 and self.vertex[2] > 0 and self.vertex[2] < 512:
                self.vinchamber = True
        except:
            self.vinchamber = False

    def set_inliers(self, initial_cluster):
        # print(initial_cluster, self.data)
        self.inliers = np.where(np.all(initial_cluster == self.data, axis = -1))[0]
        # print(self.inliers)
        # print(len(self.inliers), len(self.data))

    def calculate_length(self):
        if self.isprimary == False:
            vertice = self.vertex.copy()
            # vertice[2] = (vertice[2] - 130)*3.2*0.91
            # dados = self.data[:, :3].copy()
            # dados[:, 3] = (dados[:, 3] - 130
            normas      = norm(np.stack([vertice]*len(self.data),0) - self.data[:, :3], axis = 1)
            self.length = normas.max()
            ponto       = np.where(normas == self.length)[0]
            self.maxPos = self.data[ponto][0]
            # vp          = np.array([np.sign(self.maxPos[0])*np.abs(self.vertex[0]), 
            #                         np.sign(self.maxPos[1])*np.abs(self.vertex[1]),
            #                         -np.sign(self.maxPos[2])*np.abs(self.vertex[2])])
        else:
            a = self.data[:, :3]
            dists = distance.cdist(a, a, 'euclidean')
            self.length = np.max(dists)
            pontos = np.where(dists == self.length)[0]
            if self.data[pontos[0]][2] < self.data[pontos[1]][2]:
                self.maxPos = self.data[pontos[0]]
            else:
                self.maxPos = self.data[pontos[1]]
            self.length = -1000. # Linha extra para o ATTPC
            
            # self.maxPos = self.data[ponto]
    def get_data(self):
        return self.data
    
    def get_versor(self):
        return self.versor    
    
    def get_pb(self):
        return self.pb
        
    def get_TCharge(self):
        return self.TCharge
    
    def get_AvgCharge(self):
        return self.AvgCharge
    
    def get_vertex(self):
        return self.vertex
    
    def get_min_dist(self):
        return self.min_dist

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
    # versor, Pb = myModule.fit3D(data.tolist())
    # return np.array(versor, dtype = float), np.array(Pb, dtype = float)
    versor, Pb = utils.fit3D(data)
    return versor, Pb


        
        
        
        
        
        