from numpy import sin, cos, pi, sqrt
import random
import math
import numpy as np
import traceback

def generateWavePattern(img, dim):
    dim = dim*1.
    ## random params...
    x, y = random.randint(int(0.05*dim), int(dim-0.05*dim)), random.randint(int(0.05*dim), int(dim-0.05*dim))
    period = random.randint(int(0.03*dim), int(0.07*dim))
    width = random.randint( int(period*1.5), int(0.3*dim))
    radius = random.randint(int(0.1*dim), int(10*dim))
    length = random.randint(int(0.05*dim), int(0.5*dim))
    init_theta = random.random() * 2 * pi
    amplitude = random.randint(15, 40)

    ## calculated values
    radians = length / radius
    max_t = radians + init_theta
    num_periods = width / period
    cx, cy = int(x - radius * cos(init_theta)), int(y - radius * sin(init_theta))
    # ds/dtheta = r, so never skip a pixel with
    dtheta = 1./radius
    # probably do something to make amplitude correspond to the brightness of the image...
    # like pick 1-2 * (standard deviation in brightness / 4) or some such
    amplitude = amplitude

    overlay = np.zeros( (int(dim),int(dim)) )

    t = init_theta
    lx, ly = None, None

    # basic wave pattern
    A = [amplitude * np.sin(j * 2 *np.pi / period) for j in range(width * period)]

    a, b, c, d = 1e6, 1e6, -1e6, -1e6
    while t < max_t:
        x, y = cos(t)*radius, sin(t)*radius
        if x == lx and y == ly: continue

        for s in range(width):
            xp = int(cx + x + s * x/sqrt(x**2+y**2))
            yp = int(cy + y + s * y/sqrt(x**2+y**2))
            # to do blur / noise
            if xp >= dim or yp >= dim or xp < 0 or yp < 0: continue
            overlay[xp, yp] += A[s]
            if xp > c:
                c = xp
            if yp > d:
                d = yp
            if xp < a:
                a = xp
            if yp < b:
                b = yp

        t += dtheta
        lx = x
        ly = y
    return overlay, (a/dim,b/dim,c/dim,d/dim)