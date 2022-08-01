from numpy import sin, cos, pi, sqrt
import random
import math
import numpy as np

def fadeout(pos, total_length, m):
    if pos > total_length / 2:
        pos = total_length - pos
    
    D = max(m, total_length*0.4)
    return min( pos / D, 1 )

def generateWavePattern( img, dim ):
    dim = dim*1.
    ## random params...
    x, y = random.randint(int(0.05*dim), int(dim-0.05*dim)), random.randint(int(0.05*dim), int(dim-0.05*dim)) # where is the first pixel?
    period = random.randint(int(0.03*dim), int(0.2*dim)) # in pixels
    width = random.randint( int(max(0.1*dim, period*2)), int(min(0.6*dim, 5*period)) ) # also in pixels
    radius = random.randint(int(0.1*dim), int(10*dim)) # radius of circle whose circumference is being traced (pixels)
    length = random.randint(width, int(0.9*dim)) # along direction of wave in pixels
    init_theta = random.random() * 2 * pi # start where on circle in radians
    amplitude = random.random() # How bright is the wave (in ... units)

    ## calculated values
    img_sigma = np.std(img[:,:,0])

    radians = length / radius
    max_t = radians + init_theta
    num_periods = width / period
    cx, cy = int(x - radius * cos(init_theta)), int(y - radius * sin(init_theta))
    
    # ds/dtheta = r, so never skip a pixel with
    dtheta = 1./(radius+width*period)# / 2

    overlay = np.zeros( (int(dim),int(dim)) )

    t = init_theta
    lx, ly = None, None

    # basic wave pattern
    A = [np.sin(j * 2 *np.pi / period) for j in range(width * period)]

    a, b, c, d = 1e6, 1e6, -1e6, -1e6
    off_x, off_y = 0, 0
    whiteout_c = 0
    total_whiteout = 0
    while t < max_t:
        x, y = cos(t)*radius, sin(t)*radius
        xd = x/sqrt(x**2+y**2)
        yd = y/sqrt(x**2+y**2)

        if x == lx and y == ly: continue
        
        r = random.random()
        if r < 0.02 and whiteout_c == 0:
            # off_x += 0.75 * period * x/sqrt(x**2+y**2)
            # off_y += 0.75 * period * x/sqrt(x**2+y**2)
            whiteout_c = int(radians / dtheta * 0.2 * (random.random()/2+0.5))
            total_whiteout = whiteout_c

        lxp, lyp = None, None
        s = 0
        while s < width * period:
            xp = (cx + x + s * xd + off_x)
            yp = (cy + y + s * yd + off_y)
            
            
            ixp, iyp = int(xp), int(yp)

            # to do blur / noise
            if ixp >= dim or iyp >= dim or ixp < 0 or iyp < 0:
                break
                
            if overlay[ixp, iyp] == 0:  
                overlay[ixp, iyp] +=  img_sigma * (1+amplitude) * A[int(s)] * fadeout(int(s), width, 0.02*dim) * fadeout(t-init_theta, radians, 0.02 * dim / radius) * (1 if whiteout_c == 0 else 1 - 2*fadeout(whiteout_c, total_whiteout, total_whiteout) )

                if ixp > c:
                    c = ixp
                if iyp > d:
                    d = iyp
                if ixp < a:
                    a = ixp
                if iyp < b:
                    b = iyp
            
            # advance to the next cell in our linear projection
            if x < 0:
                dx = math.floor( xp ) - xp
            else:
                dx = math.ceil( xp ) - xp
            if y < 0:
                dy = math.floor( yp ) - yp
            else:
                dy = math.ceil( yp ) - yp
            dx /= xd
            dy /= yd
            
            if dx < dy:
                s += dx + 1e-1
            else:
                s += dy + 1e-1

        if whiteout_c > 0:
            whiteout_c -= 1
        t += dtheta
        lx = x
        ly = y
        
    return overlay, (a/dim,b/dim,c/dim,d/dim)