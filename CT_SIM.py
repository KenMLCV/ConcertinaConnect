# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:35:15 2021

@author: Kenyon
"""
import pdb

import time
import keyboard
import math
import pdb
import re
import os
import sys
import pdb
import random
import operator
import numpy as np
from PIL import Image
import cv2
import pickle
import numpy.ma as ma
import itertools
import pandas as pd
import winsound
import sounddevice as sd
import pywinauto

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, minimum_spanning_tree, breadth_first_order
from scipy.spatial import distance_matrix
from scipy.io.wavfile import write
from scipy.optimize import minimize, LinearConstraint

import CT_CFG as CFG
import CT_FAO
import CT_PP
import CT_VIS
import CT_UTIL
import CT_ML
import CT_IMP
#import CT_AUD
def test_OTA(vid_path=''):
    #pdb.set_trace()
    if(not vid_path):
        vid_path = r'.\video\segmentation_test_1.MOV'
    path, file_name = os.path.split(vid_path)
    file_name, ext = os.path.splitext(file_name)
    img = cv2.imread(r'.\frames\0.jpg',0)
    params = optimize_hough_circles(img, 5)
    contours = find_circles(img, *params)
    #test_overlay_transforms(20, 30, 40, 20, 40) 
def optimize_hough_circles(img, target_num_circles):
    
    #minDist, canny_thresh_2, accumulator_thresh, min_radius, max_radius
    np.random.seed(10)
    #x0 = np.random.randint(0, 20, 5)
    x0 = np.array([50, 20, 20, 10, 50])
    #constraint = LinearConstraint(np.ones(n_buyers), lb=n_shares, ub=n_shares)
    cons = ({'type': 'ineq', 'fun': lambda x0:  x0[3] + 10 - x0[4]}
        )
    bounds = [(10, 100) for x in x0]
    #bounds = [(10, 100), (10, 100),(10, 100),(10, 100),(10, 100)]
    #print(prices, money_available, n_shares_per_buyer, sep="\n")
    res = minimize(
        optimize_hough_circles_objective_function,
        x0=x0,
        method='COBYLA',
        callback=optimize_hough_circles_callback_function, 
        args=(img, target_num_circles,),
        constraints=cons,
        bounds=bounds,
        options={'rhobeg': 10.0}
    )
    return res.x
def optimize_hough_circles_callback_function(Xi):
    print('{}   {}   {}   {}   {}'.format(Xi[0], Xi[1], Xi[2], Xi[3], Xi[4]))
    
def optimize_hough_circles_objective_function(x0, img, target_num_circles):
    return abs(len(find_circles(img, *x0))-target_num_circles)


def find_circles(img, minDist, canny_thresh_2, accumulator_thresh, min_radius, max_radius, vis_path='find_circles'):
    
    print('{}   {}   {}   {}   {}'.format(str(minDist), str(canny_thresh_2), str(accumulator_thresh), str(min_radius), str(max_radius)))
    #img = cv2.cvtColor(CT_FAO.fetch_image(r'.\frames\0.jpg'), cv2.COLOR_RGB2BGR)
    #img = cv2.imread(r'.\frames\0.jpg',0)
    
    #img = cv2.resize(img,(300, 300), interpolation = cv2.INTER_CUBIC)
    
    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,round(minDist),
                                param1=round(canny_thresh_2),param2=round(accumulator_thresh),minRadius=round(min_radius),maxRadius=round(max_radius))
    if circles is None:
        print("Hough Circles Failure")
        return []
    else:
        print(circles.shape)
    circles = np.uint16(np.around(circles))
    
    contour_img = np.zeros(cimg.shape, np.uint8)
    #contour_img, contours, _ = CT_IMP.contour_mask(contour_img)
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,0,255),2)
        cv2.circle(contour_img,(i[0],i[1]),i[2],(255,255,255),-1)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        cv2.putText(cimg, "MD {} CT {} AT {} MIN_R {} MAX_R {}".format(str(round(minDist)), str(round(canny_thresh_2)), str(round(accumulator_thresh)), str(round(min_radius)), str(round(max_radius))), (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        
    cur_time = str(round(time.time() * 1000))
    contour_img, contours, _ = CT_IMP.contour_mask(contour_img)
    if(vis_path):
        cv2.imwrite(r'{}\{}.jpg'.format(vis_path, cur_time), cimg)
        cv2.imwrite(r'{}\{}_contour.jpg'.format(vis_path, cur_time), contour_img)
    return contours
         # Set up tracker.
    # Instead of MIL, you can also use
def multiTrack():
    major_ver, minor_ver, subminor_ver = (cv2.__version__).split('.')
    tracker_types = ['MIL','KCF', 'CSRT']
    tracker_types = ['KCF']
    tracker_type = tracker_types[0]
    
    for tt in tracker_types:
            
        # Read video
        video = cv2.VideoCapture(r'.\video\segmentation_test_1.MOV')
    
        # Exit if video not opened.
        if not video.isOpened():
            print ("Could not open video")
            sys.exit()
    
        # Read first frame.
        ok, frame = video.read()
        if not ok:
            print ('Cannot read video file')
            sys.exit()
        tracker_l = []
        bbox_d = {}
        for c in contours:
            # Define an initial bounding box
            #bbox = (i[0]-(int(i[2]*0)), i[1]-(int(i[2]*0)), (int(i[2]*2)), (int(i[2]*2)))
            #bbox = (i[0]-(int(i[2]*0)), i[1]-(int(i[2]*0)), (i[2]*5), (i[2]*5))
            #bbox = (i[0]-(int(i[2]*0.5)), i[1]-(int(i[2]*0.5)), (i[2]*0.5), (i[2]*0.5))
            bbox = cv2.boundingRect(c)
            # Uncomment the line below to select a different bounding box
            #bbox = cv2.selectROI(frame, False)
            
            if int(minor_ver) < 3:
                tracker = cv2.Tracker_create(tt)
            else:
                if tt == 'BOOSTING':
                    tracker = cv2.TrackerBoosting_create()
                if tt == 'MIL':
                    tracker = cv2.TrackerMIL_create()
                if tt == 'KCF':
                    tracker = cv2.TrackerKCF_create()
                if tt == 'TLD':
                    tracker = cv2.TrackerTLD_create()
                if tt == 'MEDIANFLOW':
                    tracker = cv2.legacy.TrackerMedianFlow_create()
                if tt == 'GOTURN':
                    tracker = cv2.TrackerGOTURN_create()
                if tt == 'MOSSE':
                    tracker = cv2.TrackerMOSSE_create()
                if tt == "CSRT":
                    tracker = cv2.TrackerCSRT_create()
            # Initialize tracker with first frame and bounding box
            ok = tracker.init(frame, bbox)
            tracker_l.append(tracker)
        frame_count = 0
        while True:
            # Read a new frame
            ok, frame = video.read()
            if not ok:
                break
            #frame = cv2.resize(frame,(300, 300), interpolation = cv2.INTER_CUBIC)
    
            
            # Start timer
            timer = cv2.getTickCount()
    
            # Update tracker
            for t_idx, t in enumerate(tracker_l):
                ok, bbox = t.update(frame)
                
                # Draw bounding box
                if ok:
                    # Tracking success
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                    bbox_d[t_idx] = bbox
                else :
                    # Tracking failure
                    bbox = bbox_d[t_idx]
                    cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    
            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    
            # Display tracker type on frame
            cv2.putText(frame, tt + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    
            # Display result
            
            #if(c%interval==0):
            #vc.set(cv2.CAP_PROP_FPS, c)
            video.set(1, frame_count)
            #cv2.waitKey(1)
            if(frame_count%20==0):
                cv2.imwrite(r'.\frames\{}_{}.jpg'.format(tt,frame_count), frame)
            frame_count = frame_count + 2
    video.release()
#test_overlay_transforms(30, 20, 20) 
def test_overlay_extraction():
    #device_img_root_path = r'G:\Team Drives\Developers\device-images\training'
    
    j_lh = cv2.cvtColor(CT_FAO.fetch_image(r'.\concertina_images\cc_jack_lh.jpg'), cv2.COLOR_RGB2BGR)
    j_lh = cv2.resize(j_lh,(300, 300), interpolation = cv2.INTER_CUBIC)
    
    j_rh = cv2.cvtColor(CT_FAO.fetch_image(r'.\concertina_images\cc_jack_rh.jpg'), cv2.COLOR_RGB2BGR)
    j_rh = cv2.resize(j_rh,(300, 300), interpolation = cv2.INTER_CUBIC)
    
    jack_collage = cv2.hconcat([j_lh, j_rh])
    
    
    j_lh_s = CT_IMP.scale(j_lh, 2, center=True)
    j_rh_s = CT_IMP.scale(j_rh, 2, center=True)
    
    
    jack_scaled = cv2.hconcat([j_lh_s, j_rh_s])
    jack_collage = cv2.vconcat([jack_collage, jack_scaled])
    
    
    j_lh = CT_IMP.color_quantize(j_lh_s, 2)[0]
    j_rh = CT_IMP.color_quantize(j_rh_s, 2)[0]
    
    
    jack_quantized = cv2.hconcat([j_lh, j_rh])
    jack_collage = cv2.vconcat([jack_collage, jack_quantized])
    
    j_lh = cv2.cvtColor(CT_IMP.cq_subtract_exterior(j_lh, 2), cv2.COLOR_GRAY2BGR)
    j_rh = cv2.cvtColor(CT_IMP.cq_subtract_exterior(j_rh, 2), cv2.COLOR_GRAY2BGR)
    
    
    jack_keys = cv2.hconcat([j_lh, j_rh])
    jack_collage = cv2.vconcat([jack_collage, jack_keys])
    
    
    j_lh, contours_lh, _ = CT_IMP.contour_mask(j_lh)
    j_rh, contours_rh, _ = CT_IMP.contour_mask(j_rh)
    
    jack_keys_contour = cv2.hconcat([j_lh, j_rh])
    jack_collage = cv2.vconcat([jack_collage, jack_keys_contour])
    
    centers_lh = CT_IMP.get_contour_centers(contours_lh)
    centers_rh = CT_IMP.get_contour_centers(contours_rh)
    
    
    j_lh_temp = np.copy(j_lh)
    j_rh_temp = np.copy(j_rh)
    for cl1 in centers_lh:
        for cl2 in centers_lh:
            cv2.line(j_lh_temp, cl1, cl2, (0,255,0), 1)
    for cr1 in centers_rh:
        for cr2 in centers_rh:
            cv2.line(j_rh_temp, cr1, cr2, (0,255,0), 1)
    
    jack_keys_fcg = cv2.hconcat([j_lh_temp, j_rh_temp])
    jack_collage = cv2.vconcat([jack_collage, jack_keys_fcg])
    
    centers_lh = CT_IMP.get_contour_centers(contours_lh)
    sorted(centers_lh, key=operator.itemgetter(1, 0))
    centers_lh.reverse()
    
    centers_rh = CT_IMP.get_contour_centers(contours_rh)
    sorted(centers_rh, key=operator.itemgetter(1, 0))
    centers_rh.reverse()
    
    dist_lh = distance_matrix(centers_lh, centers_lh)
    mst_lh = minimum_spanning_tree(dist_lh)
    bfo_lh = breadth_first_order(mst_lh, 0, directed=False)
    idx1, idx2 = np.nonzero(mst_lh.todense())
    
    j_lh_temp = np.copy(j_lh)
    
    for i in range(0, len(idx1)):
        cv2.line(j_lh_temp, centers_lh[idx1[i]], centers_lh[idx2[i]], (0,255,0), 1)
        
    dist_rh = distance_matrix(centers_rh, centers_rh)
    mst_rh = minimum_spanning_tree(dist_rh)
    bfo_rh = breadth_first_order(mst_rh, 0, directed=False)
    idx1, idx2 = np.nonzero(mst_rh.todense())
    
    j_rh_temp = np.copy(j_rh)
    
    for i in range(0, len(idx1)):
        cv2.line(j_rh_temp, centers_rh[idx1[i]], centers_rh[idx2[i]], (0,255,0), 1)
        
    cv2.circle(j_lh_temp, centers_lh[0], 10, (0,255,0), 3)
    cv2.circle(j_rh_temp, centers_rh[0], 10, (0,255,0), 3)
    
    for i in range(0, len(bfo_lh[0])):
        x, y = centers_lh[bfo_lh[0][i]]
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(j_lh_temp,str(bfo_lh[0][i]),(int(x+10), int(y-10)), font, 1,(0,255,0),2,cv2.LINE_AA)
    
    for i in range(0, len(bfo_rh[0])):
        x, y = centers_rh[bfo_rh[0][i]]
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(j_rh_temp,str(bfo_rh[0][i]),(int(x+10), int(y-10)), font, 1,(0,255,0),2,cv2.LINE_AA)
    
    
    jack_keys_mst = cv2.hconcat([j_lh_temp, j_rh_temp])
    jack_collage = cv2.vconcat([jack_collage, jack_keys_mst])
    
    lh_keys_l = [['', 'C3', '', ''],
                ['', '', 'A3', 'Ab|G#2'],
                ['F#2', 'F2', '', ''],
                ['', '', 'D2', ''],
                ['Bb2', 'B2', '', ''],
                ['', '', 'G2', 'G#|Ab1'],
                ['Eb1', 'E1', '', ''],
                ['', '', 'C1', 'C#1'],
                ['', 'A1', '', '']]
    
    rh_keys_l = [['AIR', '', '', '', ''],
                ['', 'Bb3', 'B3', '', ''],
                ['', '', '', 'G3', ''],
                ['', 'Eb2', 'E2', '', ''],
                ['', '', '', 'C2', 'C#2'],
                ['', '', 'A2', '', ''],
                ['', '', '', 'F1', 'F#1'],
                ['', '', 'D1', '', ''],
                ['', '', '', 'B1', 'Bb1'],
                ['', 'G#1', 'G1', '', '']]
    
    #CT_UTIL.map_notes_to_mst(lh_keys_l, (0,1), )
    
    #[['', 'C', 'A', 'Ab'],
    # ['F#', 'F', 'D', ''],
    # ['Bb', 'B', 'G', 'G#'],
    # ['Eb', 'E', 'C', 'C#'],
    # ['', 'A', '', '']]
    
    j_lh_temp = np.copy(j_lh)
    j_rh_temp = np.copy(j_rh)
    
    key_l_idx_map = {0: [0,1]}
    key_lh_contour_map = {}
    _, predecessors = shortest_path(csgraph=mst_lh, directed=False, indices=0, return_predecessors=True)
    
    
    key_l_idx = list.copy(key_l_idx_map[bfo_lh[0][0]])
    c2 = centers_lh[bfo_lh[0][0]]
    key_lh_contour_map[key_l_idx] = contours_lh[bfo_lh[0][0]]
    cv2.putText(j_lh_temp,lh_keys_l[key_l_idx[0]][key_l_idx[1]],(int(c2[0]+10), int(c2[1]+10)), font, 1,(0,255,0),2,cv2.LINE_AA)
    for i in range(1, len(bfo_lh[0])):
        key_l_idx = list.copy(key_l_idx_map[predecessors[bfo_lh[0][i]]])
        #print(bfo_lh[0][i], predecessors[bfo_lh[0][i]])
        c1 = centers_lh[predecessors[bfo_lh[0][i]]]
        c2 = centers_lh[bfo_lh[0][i]]
        key_lh_contour_map[key_l_idx] = contours_lh[bfo_lh[0][i]]
        x_d = c2[0] - c1[0]
        y_d = c2[1] - c1[1]
        if((abs(x_d / (y_d+0.01))) > 3):
           key_l_idx[1] = key_l_idx[1] + 1 if x_d > 0 else key_l_idx[1] - 1 
        elif(abs((y_d / (x_d+0.01))) > 3):
           key_l_idx[0] = key_l_idx[0] + 2 if y_d > 0 else key_l_idx[0] - 2
        else:
           key_l_idx[1] = key_l_idx[1] + 1 if x_d > 0 else key_l_idx[1] - 1
           key_l_idx[0] = key_l_idx[0] + 1 if y_d > 0 else key_l_idx[0] - 1
        key_l_idx_map[bfo_lh[0][i]] = key_l_idx
        #key_l_idx = key_l_idx_map[i]
        key_lh_contour_map[lh_keys_l[key_l_idx[0]][key_l_idx[1]]] = contours_lh[bfo_lh[0][i]]
        cv2.putText(j_lh_temp,lh_keys_l[key_l_idx[0]][key_l_idx[1]],(int(c2[0]+10), int(c2[1]+10)), font, 1,(0,255,0),2,cv2.LINE_AA)
        
        
    key_l_idx_map = {0: [0,0]}
    key_rh_contour_map = {}
    _, predecessors = shortest_path(csgraph=mst_rh, directed=False, indices=0, return_predecessors=True)
    
    key_l_idx = list.copy(key_l_idx_map[bfo_rh[0][0]])
    c2 = centers_rh[bfo_rh[0][0]]
    cv2.putText(j_rh_temp,rh_keys_l[key_l_idx[0]][key_l_idx[1]],(int(c2[0]+10), int(c2[1]+10)), font, 1,(0,255,0),2,cv2.LINE_AA)
    for i in range(1, len(bfo_rh[0])):
        key_l_idx = list.copy(key_l_idx_map[predecessors[bfo_rh[0][i]]])
        #print(bfo_rh[0][i], predecessors[bfo_rh[0][i]])
        c1 = centers_rh[predecessors[bfo_rh[0][i]]]
        c2 = centers_rh[bfo_rh[0][i]]
        x_d = c2[0] - c1[0]
        y_d = c2[1] - c1[1]
        if((abs(x_d / (y_d+0.01))) > 3):
           key_l_idx[1] = key_l_idx[1] + 1 if x_d > 0 else key_l_idx[1] - 1 
        elif(abs((y_d / (x_d+0.01))) > 3):
           key_l_idx[0] = key_l_idx[0] + 2 if y_d > 0 else key_l_idx[0] - 2
        else:
           key_l_idx[1] = key_l_idx[1] + 1 if x_d > 0 else key_l_idx[1] - 1
           key_l_idx[0] = key_l_idx[0] + 1 if y_d > 0 else key_l_idx[0] - 1
           
        key_l_idx_map[bfo_rh[0][i]] = key_l_idx
        
        
        #key_l_idx = key_l_idx_map[i]
        key_rh_contour_map[rh_keys_l[key_l_idx[0]][key_l_idx[1]]] = contours_rh[bfo_lh[0][i]]
        cv2.putText(j_rh_temp,rh_keys_l[key_l_idx[0]][key_l_idx[1]],(int(c2[0]+10), int(c2[1]+10)), font, 1,(0,255,0),2,cv2.LINE_AA)
    
    jack_keys_notes = cv2.hconcat([j_lh_temp, j_rh_temp])
    jack_collage = cv2.vconcat([jack_collage, jack_keys_notes])
    
    cv2.imwrite('jack_collage.jpg', jack_collage)
    
def test_audio_sampling():
    print("help")
    #sampler = CT_AUD.sampler_supervisor(j_lh_s, key_lh_contour_map, lh_keys_l, cursor=[0,1])
    #sampler.supervise()
    #sampler = CT_AUD.sampler_supervisor(j_rh_s, key_rh_contour_map, rh_keys_l, cursor=[0,0])
    #sampler.supervise()
