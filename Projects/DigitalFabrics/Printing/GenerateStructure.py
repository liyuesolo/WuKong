from GCodeGenerator import *

def buildCrossingFused(filename):
    gcode_file = open(filename, "w+")

    g_coder = GCodeGenerator(gcode_file, temperature=215)
    g_coder.writeHeader()
    bottom_left = [10, 10]
    top_right = [110, 110]
    n_row = 20
    n_col = 20
    amount = (top_right[1] - bottom_left[1]) // 50 * 3
    for i in range(1,n_row):
        y = bottom_left[0] + i * (top_right[0] - bottom_left[0]) / n_row
        g_coder.writeLine([bottom_left[1], top_right[1]], 
                            [y, y], 
                            [0.3, 0.3], amount, 600)
    for i in range(1,n_col):
        x = bottom_left[1] + i * (top_right[1] - bottom_left[1]) / n_col
        g_coder.writeLine([x, x], 
                            [bottom_left[0], top_right[0]], 
                            [0.5, 0.5], amount, 600)

    g_coder.writeFooter()
    g_coder.checkValid()
    gcode_file.close()

def closeFrame(g_coder, bottom_left, top_right, material_height):
    
    g_coder.writeLine([bottom_left[1] - 0.5, bottom_left[1]], 
                            [bottom_left[0], bottom_left[0]], 
                            [1.5, 1.5], material_height * 2.0, None, 100)
    g_coder.writeLine([bottom_left[1], top_right[1]], 
                            [bottom_left[0], bottom_left[0]], 
                            [1.5, 1.5], material_height, None, 800)
    g_coder.writeLine([top_right[1], top_right[1] + 0.5], 
                            [bottom_left[0], bottom_left[0]], 
                            [1.5, 1.5], material_height* 2.0, None, 100)
    
    g_coder.writeLine([bottom_left[1] - 0.5, bottom_left[1]], 
                            [top_right[0], top_right[0]], 
                            [1.5, 1.5], material_height * 2, None, 100)
    g_coder.writeLine([bottom_left[1], top_right[1]], 
                            [top_right[0], top_right[0]], 
                            [1.5, 1.5], material_height, None, 800)
    g_coder.writeLine([top_right[1], top_right[1] + 0.5], 
                            [top_right[0], top_right[0]], 
                            [1.5, 1.5], material_height * 2, None, 100)


    g_coder.writeLine([bottom_left[1], bottom_left[1]], 
                            [bottom_left[0] - 0.5, bottom_left[0]], 
                            [0.8, 0.8], material_height * 2, None, 100)
    g_coder.writeLine([bottom_left[1], bottom_left[1]], 
                            [bottom_left[0], top_right[0]], 
                            [0.8, 0.8], material_height, None, 800)
    g_coder.writeLine([bottom_left[1], bottom_left[1]], 
                            [top_right[0], top_right[0] + 0.5], 
                            [0.8, 0.8], material_height * 2, None, 100)

    g_coder.writeLine([top_right[1], top_right[1]], 
                            [bottom_left[0] - 0.5, bottom_left[0]], 
                            [0.8, 0.8], material_height * 2, None, 100)
    g_coder.writeLine([top_right[1], top_right[1]], 
                            [bottom_left[0], top_right[0]], 
                            [0.8, 0.8], material_height, None, 800)
    g_coder.writeLine([top_right[1], top_right[1]], 
                            [top_right[0], top_right[0] + 0.5], 
                            [0.8, 0.8], material_height * 2, None, 100)

def buildCrossingWithTunnel(filename):
    gcode_file = open(filename, "w+")

    g_coder = GCodeGenerator(gcode_file, temperature=185)
    g_coder.writeHeader()
    bottom_left = [10, 10]
    top_right = [160, 160]
    n_row = 30
    n_col = 30
    
    tunnel_size = 5

    g_coder.retract(-0.4)

    # write bottom layer in one direction
    # height = 0.2 mm speed = 300, good

    material_height = 0.2 * 0.025

    for i in range(1,n_row):
        y = bottom_left[0] + i * (top_right[0] - bottom_left[0]) / n_row
        g_coder.writeLine([bottom_left[1]-5, bottom_left[1]], 
                            [y, y], 
                            [0.2, 0.2], material_height * 2.0, None, 100, False, False)
        g_coder.writeLine([bottom_left[1], top_right[1]], 
                            [y, y], 
                            [0.2, 0.2], material_height, None, 600, False, False)
        g_coder.writeLine([top_right[1], top_right[1] + 5], 
                            [y, y], 
                            [0.2, 0.2], material_height* 2.0, None, 100, False, True)
        

    for i in range(1,n_col):
        x = bottom_left[1] + i * (top_right[1] - bottom_left[1]) / n_col 
        g_coder.writeLine([x, x], 
                            [bottom_left[0] - 5, bottom_left[0]], 
                            [0.2, 1.4], material_height* 2.0, None, 50, False, False)    
        g_coder.writeLine([x, x], 
                            [bottom_left[0], top_right[0]], 
                            [1.4, 1.4], material_height, None, 600, False, False) 
        g_coder.writeLine([x, x], 
                            [top_right[0], top_right[0]+5], 
                            [1.4, 0.2], material_height* 2.0, None, 100, False, True)   
    # add tunnel layer
    dx = (top_right[1] - bottom_left[1]) / n_col
    for i in range(1, n_row):
        y = bottom_left[0] + i * (top_right[0] - bottom_left[0]) / n_row

        loop_left = bottom_left[1]
        for j in range(1,n_col):
            tunnel_center = bottom_left[1] + j * dx
            tunnel_left = tunnel_center - tunnel_size / 2.0
            tunnel_right = tunnel_center + tunnel_size / 2.0 + 0.2

            if j == 1:
                g_coder.writeLine([loop_left, tunnel_left], 
                                [y, y], 
                                [0.5, 0.5], material_height, None, 300, False, False) 
            g_coder.writeLine([tunnel_left, tunnel_center], 
                            [y, y], 
                            [0.5, 3.2], material_height, None, 50, False, False) 
            g_coder.writeLine([tunnel_center, tunnel_right], 
                            [y, y], 
                            [3.2, 0.5], material_height, None, 100, False, False) 
            # g_coder.writeLine([tunnel_right, tunnel_right], 
            #                 [y, y], 
            #                 [0.6, 0.6], 0.1, 100, False)
            loop_left = tunnel_right

        g_coder.writeLine([loop_left, top_right[1]], 
                            [y, y], 
                            [0.5, 0.5], material_height, None, 300, False, True)

    closeFrame(g_coder, bottom_left, top_right, material_height)

    # g_coder.writeLine([top_right[1], top_right[1]], 
    #                         [top_right[0], top_right[0]], 
    #                         [0.8, 0.8], 1.0, 50)                            
        
    # for i in range(1,n_col):
    #     x = bottom_left[1] + i * (top_right[1] - bottom_left[1]) / n_col + top_right[1] + 40
    #     g_coder.writeLine([x, x], 
    #                         [bottom_left[0], top_right[0]], 
    #                         [0.4, 0.4], amount, 600)
    #     g_coder.writeLine([x, x], 
    #                         [bottom_left[0], top_right[0]], 
    #                         [0.4, 0.4], amount/2.0, 600)


    g_coder.writeFooter()
    g_coder.checkValid()
    gcode_file.close()

def buildCrossingWithTunnelWorked(filename):
    gcode_file = open(filename, "w+")

    g_coder = GCodeGenerator(gcode_file, temperature=185)
    g_coder.writeHeader()
    bottom_left = [10, 10]
    top_right = [110, 110]
    n_row = 10
    n_col = 10
    
    tunnel_size = 6

    g_coder.retract(-0.4)


    # write bottom layer in one direction
    # height = 0.2 mm speed = 300, good
    for i in range(1,n_row):
        y = bottom_left[0] + i * (top_right[0] - bottom_left[0]) / n_row
        g_coder.writeLine([bottom_left[1]-5, bottom_left[1]], 
                            [y, y], 
                            [0.2, 0.2], 1.0, 50, False, False)
        g_coder.writeLine([bottom_left[1], top_right[1]], 
                            [y, y], 
                            [0.2, 0.2], 8, 600, False, False)
        g_coder.writeLine([top_right[1], top_right[1] + 5], 
                            [y, y], 
                            [0.2, 0.2], 1.0, 50, False, True)
        

    for i in range(1,n_col):
        x = bottom_left[1] + i * (top_right[1] - bottom_left[1]) / n_col 
        g_coder.writeLine([x, x], 
                            [bottom_left[0] - 5, bottom_left[0]], 
                            [0.2, 1.4], 1, 50, False, False)    
        g_coder.writeLine([x, x], 
                            [bottom_left[0], top_right[0]], 
                            [1.4, 1.4], 8, 600, False, False) 
        g_coder.writeLine([x, x], 
                            [top_right[0], top_right[0]+5], 
                            [1.4, 0.2], 1, 50, False, True)   
    # add tunnel layer
    dx = (top_right[1] - bottom_left[1]) / n_col

    #10 -> 0.15
    amount_2layer = 0.15 / float(dx / 10)
    speed_2layer = 150
    for i in range(1, n_row):
        y = bottom_left[0] + i * (top_right[0] - bottom_left[0]) / n_row

        loop_left = bottom_left[1]
        for j in range(1,n_col):
            tunnel_center = bottom_left[1] + j * dx + 0.15
            tunnel_left = tunnel_center - tunnel_size / 2.0
            tunnel_right = tunnel_center + tunnel_size


            g_coder.writeLine([loop_left, tunnel_left], 
                            [y, y], 
                            [0.5, 0.5], amount_2layer, speed_2layer, False, False) 
            g_coder.writeLine([tunnel_left, tunnel_center], 
                            [y, y], 
                            [0.5, 2.8], amount_2layer, 70, False, False) 
            g_coder.writeLine([tunnel_center, tunnel_right], 
                            [y, y], 
                            [2.8, 0.5], amount_2layer * 2.0, speed_2layer, False, False) 
            # g_coder.writeLine([tunnel_right, tunnel_right], 
            #                 [y, y], 
            #                 [0.6, 0.6], 0.1, 100, False)
            loop_left = tunnel_right
        g_coder.writeLine([loop_left, top_right[1]], 
                            [y, y], 
                            [0.5, 0.5], amount_2layer, speed_2layer, False, True)

    g_coder.writeLine([bottom_left[1], top_right[1]], 
                            [bottom_left[0], bottom_left[0]], 
                            [1.5, 1.5], 10.0, 800)
    g_coder.writeLine([bottom_left[1], top_right[1]], 
                            [top_right[0], top_right[0]], 
                            [1.5, 1.5], 10.0, 800)

    g_coder.writeLine([bottom_left[1], bottom_left[1]], 
                            [bottom_left[0], bottom_left[0]], 
                            [0.8, 0.8], 1.0, 50)

    g_coder.writeLine([bottom_left[1], bottom_left[1]], 
                            [bottom_left[0], top_right[0]], 
                            [0.8, 0.8], 10.0, 800)
    g_coder.writeLine([top_right[1], top_right[1]], 
                            [bottom_left[0], top_right[0]], 
                            [0.8, 0.8], 10.0, 800)

    g_coder.writeLine([top_right[1], top_right[1]], 
                            [top_right[0], top_right[0]], 
                            [0.8, 0.8], 1.0, 50)                            
        
    # for i in range(1,n_col):
    #     x = bottom_left[1] + i * (top_right[1] - bottom_left[1]) / n_col + top_right[1] + 40
    #     g_coder.writeLine([x, x], 
    #                         [bottom_left[0], top_right[0]], 
    #                         [0.4, 0.4], amount, 600)
    #     g_coder.writeLine([x, x], 
    #                         [bottom_left[0], top_right[0]], 
    #                         [0.4, 0.4], amount/2.0, 600)


    g_coder.writeFooter()
    g_coder.checkValid()
    gcode_file.close()

def buildCrossingFused(filename):
    gcode_file = open(filename, "w+")

    g_coder = GCodeGenerator(gcode_file, temperature=185)
    g_coder.writeHeader()
    bottom_left = [10, 10]
    top_right = [110, 110]
    n_row = 10
    n_col = 10
    amount = (top_right[1] - bottom_left[1]) // 50 * 2
    
    tunnel_size = 6

    g_coder.retract(-0.4)
    for i in range(1,n_row):
        y = bottom_left[0] + i * (top_right[0] - bottom_left[0]) / n_row
        g_coder.writeLine([bottom_left[1]-5, bottom_left[1]], 
                            [y, y], 
                            [0.2, 0.2], 1.0, 50, False, False)
        g_coder.writeLine([bottom_left[1], top_right[1]], 
                            [y, y], 
                            [0.2, 0.2], 8, 600, False, False)
        g_coder.writeLine([top_right[1], top_right[1] + 5], 
                            [y, y], 
                            [0.2, 0.2], 1.0, 50, False, True)
        

    for i in range(1,n_col):
        x = bottom_left[1] + i * (top_right[1] - bottom_left[1]) / n_col 
        g_coder.writeLine([x, x], 
                            [bottom_left[0] - 5, bottom_left[0]], 
                            [0.2, 0.4], 1, 50, False, False)    
        g_coder.writeLine([x, x], 
                            [bottom_left[0], top_right[0]], 
                            [0.4, 0.4], 8, 600, False, False) 
        g_coder.writeLine([x, x], 
                            [top_right[0], top_right[0]+5], 
                            [0.4, 0.2], 1, 50, False, True)   


    g_coder.writeLine([bottom_left[1], top_right[1]], 
                            [bottom_left[0], bottom_left[0]], 
                            [0.6, 0.6], 10.0, 800)
    g_coder.writeLine([bottom_left[1], top_right[1]], 
                            [top_right[0], top_right[0]], 
                            [0.6, 0.6], 10.0, 800)

    # g_coder.writeLine([bottom_left[1], bottom_left[1]], 
    #                         [bottom_left[0], bottom_left[0]], 
    #                         [0.8, 0.8], 1.0, 50)

    g_coder.writeLine([bottom_left[1], bottom_left[1]], 
                            [bottom_left[0], top_right[0]], 
                            [0.8, 0.8], 10.0, 800)
    g_coder.writeLine([top_right[1], top_right[1]], 
                            [bottom_left[0], top_right[0]], 
                            [0.8, 0.8], 10.0, 800)

    # g_coder.writeLine([top_right[1], top_right[1]], 
    #                         [top_right[0], top_right[0]], 
    #                         [0.8, 0.8], 1.0, 50)                            
    

    g_coder.writeFooter()
    g_coder.checkValid()
    gcode_file.close()

def buildTestTunnelScene(filename):
    gcode_file = open(filename, "w+")

    g_coder = GCodeGenerator(gcode_file, temperature=215)
    g_coder.writeHeader()

    y = 15
    g_coder.writeLine([10, 40], [y, y], [0.25, 0.25], 1.5, 800)
    g_coder.writeLine([10, 23], [y, y], [0.5, 0.5], 0.5, 800, False)
    g_coder.writeLine([23, 25], [y, y], [0.5, 2.0], 0.2, 50, False)
    g_coder.writeLine([25, 27], [y, y], [2.0, 0.5], 0.2, 150, False)
    g_coder.writeLine([27, 27], [y, y], [0.5, 0.5], 0.1, 100, False)
    g_coder.writeLine([27, 40], [y, y], [0.5, 0.5], 0.5, 800, False)

    y = 25
    g_coder.writeLine([10, 40], [y, y], [0.25, 0.25], 1.5, 800)

    ## this one works
    y = 35
    g_coder.writeLine([10, 40], [y, y], [0.25, 0.25], 1.5, 800)
    g_coder.writeLine([10, 23], [y, y], [0.5, 0.5], 0.5, 800, False)
    g_coder.writeLine([23, 25], [y, y], [0.5, 3.0], 0.2, 50, False)
    g_coder.writeLine([25, 27], [y, y], [3.0, 0.5], 0.2, 150, False)
    g_coder.writeLine([27, 27], [y, y], [0.5, 0.5], 0.1, 100, False)
    g_coder.writeLine([27, 40], [y, y], [0.5, 0.5], 0.5, 800, False)

    y = 45
    g_coder.writeLine([10, 40], [y, y], [0.25, 0.25], 1.5, 800)

    y = 55
    g_coder.writeLine([10, 40], [y, y], [0.25, 0.25], 1.5, 800)
    g_coder.writeLine([10, 23], [y, y], [0.5, 0.5], 0.5, 800, False)
    g_coder.writeLine([23, 25], [y, y], [0.5, 2.0], 0.2, 100, False)
    g_coder.writeLine([25, 27], [y, y], [2.0, 0.5], 0.2, 150, False)
    g_coder.writeLine([27, 27], [y, y], [0.5, 0.5], 0.1, 100, False)
    g_coder.writeLine([27, 40], [y, y], [0.5, 0.5], 0.5, 800, False)

    y = 65
    g_coder.writeLine([10, 40], [y, y], [0.25, 0.25], 1.5, 800)

    y = 75
    g_coder.writeLine([10, 40], [y, y], [0.25, 0.25], 1.5, 800)
    g_coder.writeLine([10, 23], [y, y], [0.4, 0.4], 0.5, 800, False)
    g_coder.writeLine([23, 27], [y, y], [0.5, 3.0], 0.5, 20, False)
    g_coder.writeLine([27, 40], [y, y], [0.4, 0.4], 0.5, 800, False)

    g_coder.writeFooter()
    g_coder.checkValid()
    gcode_file.close()

# buildTestTunnelScene("tunnel.gcode")
buildCrossingWithTunnel("tunnel_fabric.gcode")
# buildCrossingFused("tunnel_fused.gcode")