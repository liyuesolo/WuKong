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

def buildCrossingWithTunnel(filename):
    gcode_file = open(filename, "w+")

    g_coder = GCodeGenerator(gcode_file, temperature=215)
    g_coder.writeHeader()
    bottom_left = [10, 10]
    top_right = [60, 60]
    n_row = 10
    n_col = 10
    amount = (top_right[1] - bottom_left[1]) // 50 * 2
    
    tunnel_size = 6

    g_coder.retract(-0.4)
    for i in range(1,n_row):
        y = bottom_left[0] + i * (top_right[0] - bottom_left[0]) / n_row
        g_coder.writeLine([bottom_left[1], top_right[1]], 
                            [y, y], 
                            [0.2, 0.2], amount, 600)

    for i in range(1,n_col):
        x = bottom_left[1] + i * (top_right[1] - bottom_left[1]) / n_col 
        g_coder.writeLine([x, x], 
                            [bottom_left[0], top_right[0]], 
                            [0.4, 0.4], amount, 600)    

    # add tunnel layer
    dx = (top_right[1] - bottom_left[1]) / n_col
    for i in range(1,n_row):
        y = bottom_left[0] + i * (top_right[0] - bottom_left[0]) / n_row

        loop_left = bottom_left[1]
        for i in range(1,n_col):
            tunnel_center = bottom_left[1] + i * dx
            tunnel_left = tunnel_center - tunnel_size / 2.0
            tunnel_right = tunnel_center + tunnel_size / 2.0

            g_coder.writeLine([loop_left, tunnel_left], 
                            [y, y], 
                            [0.4, 0.4], amount/2.0, 800, False)
            g_coder.writeLine([tunnel_left, tunnel_center], 
                            [y, y], 
                            [0.4, 2.2], 0.2, 50, False)
            g_coder.writeLine([tunnel_center, tunnel_right], 
                            [y, y], 
                            [2.2, 0.4], 0.2, 150, False)
            g_coder.writeLine([tunnel_right, tunnel_right], 
                            [y, y], 
                            [0.4, 0.4], 0.1, 100, False)
            loop_left = tunnel_right
        g_coder.writeLine([tunnel_right, top_right[1]], 
                            [y, y], 
                            [0.4, 0.4], amount/2.0, 800)


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