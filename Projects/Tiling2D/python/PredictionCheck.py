import os
from functools import cmp_to_key


from joblib import Parallel, delayed


from requests import options
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import math
import numpy as np
import tensorflow as tf
from model import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
import scipy
from scipy.optimize import check_grad
from tactile import IsohedralTiling, tiling_types, EdgeShape, mul, Point
import dearpygui.dearpygui as dpg

def loadDataSplitTest(n_tiling_params, filename, shuffle = True, ignore_unconverging_result = True):
    all_data = []
    all_label = [] 
    
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")[:]]
        if (ignore_unconverging_result):
            if (item[-1] > 1e-6 or math.isnan(item[-1])):
                continue
            if (item[-5] < 1e-5 or item[-5] > 10):
                continue
        data = item[0:n_tiling_params]
        for i in range(2):
            data.append(item[n_tiling_params+i])
        data.append(2.0 * item[n_tiling_params+2])
        
        label = item[n_tiling_params+3:n_tiling_params+7]
        
        all_data.append(data)
        all_label.append(label)
        
    print("#valid data:{}".format(len(all_data)))
    # exit(0)
    start = 0
    end = -1
    all_data = np.array(all_data[start:]).astype(np.float32)
    all_label = np.array(all_label[start:]).astype(np.float32) 
    
    # all_data = np.array(all_data).astype(np.float32)
    # all_label = np.array(all_label).astype(np.float32)
    
    indices = np.arange(all_data.shape[0])
    if (shuffle):
        np.random.shuffle(indices)
    
    all_data = all_data[indices]
    all_label = all_label[indices]
    
    return all_data, all_label

@tf.function
def testStep(n_tiling_params, lambdas, model):
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(lambdas)
        with tf.GradientTape() as tape:
            tape.watch(lambdas)
            
            elastic_potential = model(lambdas, training=False)
            dedlambda = tape.gradient(elastic_potential, lambdas)
            batch_dim = elastic_potential.shape[0]
            stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
            de_dp = tf.slice(dedlambda, [0, 0], [batch_dim, n_tiling_params])
    dstress_dp = tape_outer.batch_jacobian(stress, lambdas)[:, :, 0:n_tiling_params]
    del tape
    del tape_outer
    return dstress_dp, stress, de_dp, elastic_potential

def make_random_tiling(IH, params):
    # Construct a tiling
    # tiling = IsohedralTiling(random.choice(tiling_types))
    tiling = IsohedralTiling(IH)

    # Randomize the tiling vertex parameters
    ps = tiling.parameters
    for i in range(tiling.num_parameters):
        ps[i] = params[i]
    tiling.parameters = ps

    edges = []
    for shp in tiling.edge_shapes:
        ej = []
        ej.append(Point(0, 0))
        ej.append(Point(0.25, 0))
        ej.append(Point(0.75, 0))
        ej.append(Point(1, 0))
        if shp == EdgeShape.I:
            # Must be a straight line.
            pass
        elif shp == EdgeShape.J:
            # Anything works for J
            ej[2] = Point(1.0 - ej[1].x, ej[1].y)
            
        elif shp == EdgeShape.S:
            # 180-degree rotational symmetry
            ej[2] = Point(1.0 - ej[1].x, -ej[1].y)
            
        elif shp == EdgeShape.U:
            # Symmetry after reflecting/flipping across length.
            ej[1] = Point(ej[1].x, 0.0)
            ej[2] = Point(ej[2].x, 0.0)
            
        edges.append( ej )

    return tiling, edges
    
def getPolygons(IH, params):
    tx = 0
    ty = 0
    scale = 100
    tiling, edges = make_random_tiling(IH, params)
    

    ST = [scale, 0.0, tx, 0.0, scale, ty]
    polygons = []
    for i in tiling.fill_region_bounds( -5, -5, 10, 10 ):
        T = mul( ST, i.T )
        
        start = True
        polygon = []
        
        for si in tiling.shapes:
            
            S = mul( T, si.T )
            # Make the edge start at (0,0)
            seg = [ mul( S, Point(0., 0.)) ]

            if si.shape != EdgeShape.I:
                # The edge isn't just a straight line.
                ej = edges[ si.id ]
                seg.append( mul( S, ej[0] ) )
                seg.append( mul( S, ej[1] ) )

            # Make the edge end at (1,0)
            seg.append( mul( S, Point(1., 0.)) )

            if si.rev:
                seg.reverse()

            if start:
                start = False
                polygon.append([seg[0].x, seg[0].y])
            if len(seg) == 2:
                polygon.append([seg[1].x, seg[1].y])
            else:
                polygon.append([seg[1].x, seg[1].y])
                polygon.append([seg[2].x, seg[2].y])
                polygon.append([seg[3].x, seg[3].y])

        
        polygons.append(polygon)
    
    return polygons

def explorer():
    update = False
    IH = 21
    current_dir = os.path.dirname(os.path.realpath(__file__))

    if IH == 50:
        n_tiling_params = 2
        save_path = os.path.join(current_dir, 'Models/' + str(81) + "/")
        model = buildSingleFamilyModelSeparateTilingParams(n_tiling_params)
        model.load_weights(save_path + "IH50" + '.tf')
        range_t0 = [0.1, 0.3]
        range_t1 = [0.25, 0.75]
        x0 = np.array([0.2, 0.5])
        params = [0.2, 0.5]
    elif IH == 21:
        n_tiling_params = 2
        save_path = os.path.join(current_dir, 'Models/' + str(221) + "/")
        model = buildSingleFamilyModel3Strain(n_tiling_params)
        model.load_weights(save_path + "IH21" + '.tf')
        range_t0 = [0.1, 0.2]
        range_t1 = [0.5, 0.8]
        x0 = np.array([0.15, 0.55])
        params = [0.15, 0.55]
    
    polygons = getPolygons(IH, params)

    query_points = []
    range_strain = [-0.3, 0.5]
    n_sp_strain = 200
    delta_strain = (range_strain[1] - range_strain[0]) / float(n_sp_strain)
    strain_xx = []
    for i in range(n_sp_strain):
        strain = range_strain[1] - delta_strain * float(i)
        strain_xx.append(strain)
        query_points.append(np.hstack((x0, np.array([strain, 0.06, 0.0004, 0.0004]))))
        
    
    model_input = tf.convert_to_tensor(query_points)

    _, _, de_dp, psi = testStep(n_tiling_params, model_input, model)
    

    def t1Callback(sender):
        update = True
        for pt in query_points:
            pt[0] = dpg.get_value(sender)
        t1 = float(dpg.get_value(sender))
        
        polygons = getPolygons(IH, [t1, query_points[0][1]])
        model_input = tf.convert_to_tensor(query_points)
        _, _, _, psi = testStep(n_tiling_params, model_input, model)
        if update:
            dpg.set_value('plot', [strain_xx, psi])
            dpg.delete_item(item="draw")
            dpg.add_draw_layer(tag="draw", parent="structure")
            
            for polygon in polygons:
                for i in range(len(polygon) - 1):
                    vi, vj = polygon[i], polygon[i+1]
                    dpg.draw_line((vi[0], vi[1]), (vj[0], vj[1]), color=(255, 0, 0, 255), thickness=3, parent="draw")
            
            
        update = False 
        
    
    def t2Callback(sender):
        update = True
        for pt in query_points:
            pt[1] = dpg.get_value(sender)
        t2 = float(dpg.get_value(sender))
        
        polygons = getPolygons(IH, [query_points[0][0], t2])
        model_input = tf.convert_to_tensor(query_points)
        _, _, _, psi = testStep(n_tiling_params, model_input, model)
        if update:
            dpg.set_value('plot', [strain_xx, psi])
            dpg.delete_item(item="draw")
            dpg.add_draw_layer(tag="draw", parent="structure")
            for polygon in polygons:
                for i in range(len(polygon) - 1):
                    vi, vj = polygon[i], polygon[i+1]
                    dpg.draw_line((vi[0], vi[1]), (vj[0], vj[1]), color=(255, 0, 0, 255), thickness=3, parent="draw")
        update = False 
    def strainyyCallback(sender):
        for pt in query_points:
            pt[3] = dpg.get_value(sender)
        model_input = tf.convert_to_tensor(query_points)
        _, _, _, psi = testStep(n_tiling_params, model_input, model)
        dpg.set_value('plot', [strain_xx, psi])
    
    def strainxyCallback(sender):
        for pt in query_points:
            pt[4] = dpg.get_value(sender)
            pt[5] = dpg.get_value(sender)
        model_input = tf.convert_to_tensor(query_points)
        _, _, _, psi = testStep(n_tiling_params, model_input, model)
        dpg.set_value('plot', [strain_xx, psi])

    dpg.create_context()
    dpg.create_viewport(title='Neural Constitutive Model',width=3700, height=2200)
    dpg.setup_dearpygui()
    dpg.set_global_font_scale(2.5)
    x = []
    y = []
    for polygon in polygons:
        for i in range(len(polygon) - 1):
            vi, vj = polygon[i], polygon[i+1]
            x.append(vi)
            y.append(vj)

    with dpg.window(label="Tiling Explorer"):
        with dpg.group(horizontal=True):
            dpg.add_text("Tiling Parameters")
            dpg.add_slider_float(default_value=0.5 * (range_t0[0] + range_t0[1]), min_value = range_t0[0], max_value=range_t0[1],width=200.0, height=500.0, callback= t1Callback)
            dpg.add_slider_float(default_value=0.5 * (range_t1[0] + range_t1[1]), min_value = range_t1[0], max_value=range_t1[1],width=200.0, height=500.0, callback= t2Callback)
            dpg.add_text("Strain yy Strain xy")
            dpg.add_slider_float(default_value=0.01, min_value = -0.3, max_value=0.5,width=200.0, height=500.0, callback= strainyyCallback)
            dpg.add_slider_float(default_value=0.001, min_value = -0.01, max_value=0.01,width=200.0, height=500.0, callback= strainxyCallback)
        with dpg.theme(tag="plot_theme"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (150, 255, 0), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 10.0, category=dpg.mvThemeCat_Plots)
                
                # dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Diamond, category=dpg.mvThemeCat_Plots)
                # dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 7, category=dpg.mvThemeCat_Plots)
        with dpg.group(horizontal=True):
            with dpg.plot(label="Line Series", height=2000, width=2600):
                # optionally create legend
                dpg.add_plot_legend()

                # REQUIRED: create x and y axes
                dpg.add_plot_axis(dpg.mvXAxis, label="strain_xx")
                dpg.add_plot_axis(dpg.mvYAxis, label="energy density", tag="y_axis")

                # series belong to a y axis
                dpg.add_line_series(strain_xx, psi, label="enery", parent="y_axis", tag="plot")
                dpg.bind_item_theme("plot", "plot_theme")
            with dpg.drawlist(width=1000, height=1000, tag="structure"):
                dpg.add_draw_layer(tag="draw", parent="structure")
                for polygon in polygons:
                    for i in range(len(polygon) - 1):
                        vi, vj = polygon[i], polygon[i+1]
                        dpg.draw_line((vi[0], vi[1]), (vj[0], vj[1]), color=(255, 0, 0, 255), thickness=3, parent="draw")
                
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
    return

def checkoutTestingData():
    

    update = False
    IH = 21
    current_dir = os.path.dirname(os.path.realpath(__file__))

    if IH == 50:
        n_tiling_params = 2
        save_path = os.path.join(current_dir, 'Models/' + str(81) + "/")
        model = buildSingleFamilyModelSeparateTilingParams(n_tiling_params)
        model.load_weights(save_path + "IH50" + '.tf')
        range_t0 = [0.1, 0.3]
        range_t1 = [0.25, 0.75]
        x0 = np.array([0.2, 0.5])
        params = [0.2, 0.5]
    elif IH == 21:
        n_tiling_params = 2
        save_path = os.path.join(current_dir, 'Models/' + str(221) + "/")
        model = buildSingleFamilyModel3Strain(n_tiling_params)
        model.load_weights(save_path + "IH21" + '.tf')
        range_t0 = [0.1, 0.2]
        range_t1 = [0.5, 0.8]
        x0 = np.array([0.15, 0.55])
        params = [0.15, 0.55]
        full_data = "/home/yueli/Documents/ETH/SandwichStructure/Server/all_data_IH21_shuffled.txt"  
        data_all, label_all = loadDataSplitTest(n_tiling_params, full_data, False, True)
        five_percent = int(len(data_all) * 0.05)
        
        train_data =  data_all[:-five_percent]
        train_label =  label_all[:-five_percent]

        validation_data = data_all[-five_percent:]
        validation_label = label_all[-five_percent:]

        grad_loss, e_loss, sigma, energy = testStep(n_tiling_params, tf.convert_to_tensor(validation_data), model)
    
    check_test_set = False

    polygons = getPolygons(IH, params)

    query_points = []
    range_strain = [-0.3, 0.5]
    n_sp_strain = 200
    delta_strain = (range_strain[1] - range_strain[0]) / float(n_sp_strain)
    strain_xx = []
    for i in range(n_sp_strain):
        strain = range_strain[1] - delta_strain * float(i)
        strain_xx.append(strain)
        query_points.append(np.hstack((x0, np.array([strain, 0.06, 0.0004]))))
        
    
    model_input = tf.convert_to_tensor(query_points)

    _, _, de_dp, psi = testStep(n_tiling_params, model_input, model)
    

    def t1Callback(sender):
        update = True
        for pt in query_points:
            pt[0] = dpg.get_value(sender)
        t1 = float(dpg.get_value(sender))
        
        polygons = getPolygons(IH, [t1, query_points[0][1]])
        model_input = tf.convert_to_tensor(query_points)
        _, _, _, psi = testStep(n_tiling_params, model_input, model)
        if update:
            dpg.set_value('plot', [strain_xx, psi])
            dpg.delete_item(item="draw")
            dpg.add_draw_layer(tag="draw", parent="structure")
            
            for polygon in polygons:
                for i in range(len(polygon) - 1):
                    vi, vj = polygon[i], polygon[i+1]
                    dpg.draw_line((vi[0], vi[1]), (vj[0], vj[1]), color=(255, 0, 0, 255), thickness=3, parent="draw")
            
            
        update = False 
        
    
    def t2Callback(sender):
        update = True
        for pt in query_points:
            pt[1] = dpg.get_value(sender)
        t2 = float(dpg.get_value(sender))
        
        polygons = getPolygons(IH, [query_points[0][0], t2])
        model_input = tf.convert_to_tensor(query_points)
        _, _, _, psi = testStep(n_tiling_params, model_input, model)
        if update:
            dpg.set_value('plot', [strain_xx, psi])
            dpg.delete_item(item="draw")
            dpg.add_draw_layer(tag="draw", parent="structure")
            for polygon in polygons:
                for i in range(len(polygon) - 1):
                    vi, vj = polygon[i], polygon[i+1]
                    dpg.draw_line((vi[0], vi[1]), (vj[0], vj[1]), color=(255, 0, 0, 255), thickness=3, parent="draw")
        update = False 
    def strainyyCallback(sender):
        for pt in query_points:
            pt[3] = dpg.get_value(sender)
        model_input = tf.convert_to_tensor(query_points)
        _, _, _, psi = testStep(n_tiling_params, model_input, model)
        dpg.set_value('plot', [strain_xx, psi])
    
    def strainxyCallback(sender):
        for pt in query_points:
            pt[4] = dpg.get_value(sender)
        model_input = tf.convert_to_tensor(query_points)
        _, _, _, psi = testStep(n_tiling_params, model_input, model)
        dpg.set_value('plot', [strain_xx, psi])

    dpg.create_context()
    dpg.create_viewport(title='Neural Constitutive Model',width=3700, height=2200)
    dpg.setup_dearpygui()
    dpg.set_global_font_scale(2.5)
    x = []
    y = []
    for polygon in polygons:
        for i in range(len(polygon) - 1):
            vi, vj = polygon[i], polygon[i+1]
            x.append(vi)
            y.append(vj)

    def plotTilingExplorer():
        if dpg.does_item_exist("param"):
            dpg.delete_item(item="param")
        if dpg.does_item_exist("explore"):
            dpg.delete_item(item="explore")
        with dpg.group(horizontal=True, tag = 'param', parent='base_window'):
            dpg.add_text("Tiling Parameters")
            dpg.add_slider_float(default_value=0.5 * (range_t0[0] + range_t0[1]), min_value = range_t0[0], max_value=range_t0[1],width=200.0, height=500.0, callback= t1Callback)
            dpg.add_slider_float(default_value=0.5 * (range_t1[0] + range_t1[1]), min_value = range_t1[0], max_value=range_t1[1],width=200.0, height=500.0, callback= t2Callback)
            dpg.add_text("Strain yy Strain xy")
            dpg.add_slider_float(default_value=0.01, min_value = -0.3, max_value=0.5,width=200.0, height=500.0, callback= strainyyCallback)
            dpg.add_slider_float(default_value=0.001, min_value = -0.3, max_value=0.3,width=200.0, height=500.0, callback= strainxyCallback)

        
                # dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Diamond, category=dpg.mvThemeCat_Plots)
                # dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 7, category=dpg.mvThemeCat_Plots)
        with dpg.group(horizontal=True, tag='explore', parent='base_window'):
            with dpg.plot(label="Line Series", height=2000, width=2600):
                # optionally create legend
                dpg.add_plot_legend()

                # REQUIRED: create x and y axes
                dpg.add_plot_axis(dpg.mvXAxis, label="strain_xx")
                dpg.add_plot_axis(dpg.mvYAxis, label="energy density", tag="y_axis")

                # series belong to a y axis
                dpg.add_line_series(strain_xx, psi, label="enery", parent="y_axis", tag="plot")
                dpg.bind_item_theme("plot", "plot_theme")
            with dpg.drawlist(width=1000, height=1000, tag="structure"):
                dpg.add_draw_layer(tag="draw", parent="structure")
                for polygon in polygons:
                    for i in range(len(polygon) - 1):
                        vi, vj = polygon[i], polygon[i+1]
                        dpg.draw_line((vi[0], vi[1]), (vj[0], vj[1]), color=(255, 0, 0, 255), thickness=3, parent="draw")

    

    def predictionChecker():
        if dpg.does_item_exist("param"):
            dpg.delete_item(item="param")
        if dpg.does_item_exist("explore"):
            dpg.delete_item(item="explore")
        
        
        elastic_potential = model(tf.convert_to_tensor(validation_data), training = False)

        potential_gt = validation_label[:, -1] # last entry is the potential
        # potential_pred = energy.numpy() # prediction 
        potential_pred = elastic_potential.numpy() #identical to above
        indices = [i for i in range(len(potential_gt))]
        
        def compare_energy(i, j):
            return potential_gt[i] - potential_gt[j]
        indices_sorted = sorted(indices, key=cmp_to_key(compare_energy))
        with dpg.group(horizontal=True, tag='explore', parent='base_window'):
            with dpg.plot(label="Energy", height=1000, width=1800):
                dpg.add_plot_legend()
                # dpg.add_plot_legend(location=dpg.mvPlot_Location_East)

                # REQUIRED: create x and y axes
                dpg.add_plot_axis(dpg.mvXAxis, label="sample idx")
                dpg.add_plot_axis(dpg.mvYAxis, label="energy density", tag="y_axis")

                # series belong to a y axis
                dpg.add_line_series(indices, potential_pred[indices_sorted], label="energy_pred", parent="y_axis", tag="plot")
                dpg.add_line_series(indices, potential_gt[indices_sorted], label="enery_gt", parent="y_axis", tag="plot2")
                
                dpg.bind_item_theme("plot", "plot_theme")
                dpg.bind_item_theme("plot2", "plot_theme_gt")
            with dpg.plot(label="Stress xx", height=1000, width=1800):
                dpg.add_plot_legend()
                # dpg.add_plot_legend(location=dpg.mvPlot_Location_East)

                # REQUIRED: create x and y axes
                dpg.add_plot_axis(dpg.mvXAxis, label="sample idx")
                dpg.add_plot_axis(dpg.mvYAxis, label="stress xx", tag="y_axis2")

                # series belong to a y axis
                dpg.add_line_series(indices, potential_pred[indices_sorted], label="strain_xx_pred", parent="y_axis2", tag="plot3")
                dpg.add_line_series(indices, potential_gt[indices_sorted], label="strain_xx_gt", parent="y_axis2", tag="plot4")
                
                dpg.bind_item_theme("plot3", "plot_theme")
                dpg.bind_item_theme("plot4", "plot_theme_gt")
        with dpg.group(horizontal=True, tag='explore2', parent='base_window'):
            with dpg.plot(label="shear xy", height=1000, width=1800):
                dpg.add_plot_legend()
                # dpg.add_plot_legend(location=dpg.mvPlot_Location_East)

                # REQUIRED: create x and y axes
                dpg.add_plot_axis(dpg.mvXAxis, label="sample idx")
                dpg.add_plot_axis(dpg.mvYAxis, label="shear", tag="y_axis3")

                # series belong to a y axis
                dpg.add_line_series(indices, potential_pred[indices_sorted], label="shear_xy_pred", parent="y_axis3", tag="plot5")
                dpg.add_line_series(indices, potential_gt[indices_sorted], label="shear_xy_gt", parent="y_axis3", tag="plot6")
                
                dpg.bind_item_theme("plot5", "plot_theme")
                dpg.bind_item_theme("plot6", "plot_theme_gt")
            with dpg.plot(label="stress yy", height=1000, width=1800):
                dpg.add_plot_legend()
                # dpg.add_plot_legend(location=dpg.mvPlot_Location_East)

                # REQUIRED: create x and y axes
                dpg.add_plot_axis(dpg.mvXAxis, label="sample idx")
                dpg.add_plot_axis(dpg.mvYAxis, label="energy density", tag="y_axis4")

                # series belong to a y axis
                dpg.add_line_series(indices, potential_pred[indices_sorted], label="strain_yy_pred", parent="y_axis4", tag="plot7")
                dpg.add_line_series(indices, potential_gt[indices_sorted], label="strain_yy_gt", parent="y_axis4", tag="plot8")
                
                dpg.bind_item_theme("plot7", "plot_theme")
                dpg.bind_item_theme("plot8", "plot_theme_gt")  
            

    def changeTest(sender):
        nonlocal check_test_set
        check_test_set = dpg.get_value(sender)
        if check_test_set == True:
            
            predictionChecker()
        else:
            plotTilingExplorer()

    with dpg.window(label="Tiling Explorer", tag="base_window", height=2000, width=3600):
        with dpg.theme(tag="plot_theme"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (150, 255, 0), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 10.0, category=dpg.mvThemeCat_Plots)
        with dpg.theme(tag="plot_theme_gt"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (150, 20, 255), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 10.0, category=dpg.mvThemeCat_Plots)
        
        dpg.add_checkbox(label="Check Test Data", callback=changeTest)
        
                
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
    return

if __name__ == "__main__":
    checkoutTestingData()