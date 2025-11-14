##
## NOTE: smoothing below controls how closely the polylines are matched
##

import numpy as np
import xml.etree.ElementTree as ET
from scipy.interpolate import splprep, splev
import re
import scipy.interpolate as si
import matplotlib.pyplot as plt
import argparse

def load_svg_polylines(svg_file):
    """
    Loads all polylines from an SVG file where polylines are defined as <path d="..."/>.

    Returns:
        List of polylines. Each polyline is a list of (x, y) tuples.
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # Namespace handling (in case SVG uses it)
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    polylines = []

    # Find all <path> elements (works with or without namespace)
    for path in root.findall('.//{http://www.w3.org/2000/svg}path') + root.findall('.//path'):
        d = path.get('d')
        if not d:
            continue

        # Extract all commands with their coordinates
        # Supports only absolute M and L commands for simplicity
        matches = re.findall(r'([ML])\s*([-\d\.]+)[ ,]+([-\d\.]+)', d, re.IGNORECASE)
        if not matches:
            continue
        
        polyline = [(float(x), float(y)) for cmd, x, y in matches]
        polylines.append(polyline)

    return polylines

def scipy_bspline(cv, degree=3):
    """ cv:       Array of control vertices
        degree:   Curve degree
    """
    periodic = 0 ## NOTE: unclear how to use this for cyclic path?
    count = cv.shape[0]

    degree = np.clip(degree, 1, count-1)
    kv = np.clip(np.arange(count+degree+1)-degree, 0, count-degree)

    max_param = count - (degree * (1-periodic))
    spline = si.BSpline(kv, cv, degree)
    return spline, max_param

def bspline_to_bezier(cv):
    cv_len = cv.shape[0]
    # print(cv.shape)
    assert cv_len >= 4, "Provide at least 4 control vertices"
    spline, max_param = scipy_bspline(cv, degree=3)
    for i in range(1, max_param):
        spline = si.insert(i, spline, 2)
    return spline.c[:3 * max_param + 1]

def bspline_to_svg_path(tck,cyclic):
    degree = tck[2]
    knots, coeffs, k = tck
    xy_coeffs=[]
    for i in range(0,len(coeffs[0])):
        xy_coeffs.append([coeffs[0][i],coeffs[1][i]])
    xy_coeffs = np.array(xy_coeffs)
    # print(xy_coeffs)
    bezier = bspline_to_bezier(xy_coeffs)
    # if cyclic:
        # tmp = []
        # for i in range(2,len(bezier)):
            # tmp.append(bezier[i])
        # bezier = tmp
    
    path = f'M {bezier[0][0]} {bezier[0][1]} '
    for i in range(1, len(bezier) - 1, 3):
        v1, v2, v = bezier[i:i+3]
        path += f'C {v1[0]} {v1[1]} {v2[0]},{v2[1]} {v[0]},{v[1]} '
    return f'<path d="{path.strip()}" fill="none" stroke="blue" stroke-width="0.1" />'

def wrap_svg(paths, width=None, height=None, margin=10):
    # Determine bounding box from all paths (roughly)
    all_x = []
    all_y = []
    for path in paths:
        import re
        numbers = list(map(float, re.findall(r'[-+]?\d*\.\d+|\d+', path)))
        xs = numbers[::2]
        ys = numbers[1::2]
        all_x.extend(xs)
        all_y.extend(ys)
    min_x, max_x = min(all_x)-margin, max(all_x)+margin
    min_y, max_y = min(all_y)-margin, max(all_y)+margin
    width = width or (max_x - min_x)
    height = height or (max_y - min_y)

    header = """<svg
   height="70mm"
   id="svg11"
   inkscape:version="1.4.2 (f4327f4, 2025-05-13)"
   sodipodi:docname="out_smiley.svg"
   version="1.1"
   viewBox="0 0 70.000002 69.999998"
   width="70mm"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:svg="http://www.w3.org/2000/svg">"""
    footer = '</svg>'

    body = "\n".join(paths)
    return f'{header}\n{body}\n{footer}'

def polyline_to_svg_path(poly):
    path = f'M {poly[0][0]} {poly[0][1]} '
    for i in range(1, len(poly)):
        v = poly[i]
        path += f'L {v[0]},{v[1]} '
    return f'<path d="{path.strip()}" fill="none" stroke="blue" stroke-width="0.1" />'


# ----------------------------
# Cubic spline approximation
# ----------------------------
def remove_duplicates(points):
    """Remove consecutive duplicate points."""
    return [points[i] for i in range(len(points)) if i == 0 or points[i] != points[i-1]]

## NOTE: unused, for debugging, resamples and displays spline
def polyline_to_cubic_spline(points, smoothing=0.0, num_points=50):
    """
    Approximate a polyline with a cubic spline.
    Returns a list of (x, y) points.
    """
    points = remove_duplicates(points)
    if len(points) < 2:
        return points
    x, y = zip(*points)
    plt.plot(x, y, 'b')
    cyclic = (x[0] == x[-1] and y[0] == y[-1]) ## NOTE always false, see path loading
    tck, u = splprep([x, y], s=smoothing, per=cyclic)
    # print(len(points), " vs ",len(tck[0]))
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    plt.plot(x_new, y_new, 'g')
    return list(zip(x_new, y_new))

def polyline_to_tck_cubic_spline(points, smoothing, cyclic, extremity_len=4, extremity_w=0.1):
    """
    Approximate a polyline with a cubic spline.
    Returns a list of (x, y) points.
    """
    points = remove_duplicates(points)
    x, y = zip(*points)
    weights = np.full(len(points),1.0)
    if len(points) > 2*extremity_len and not cyclic:
        for i in range(0,4):
            weights[i]=extremity_w
            weights[-1-i]=extremity_w
    tck, u = splprep([x, y], s=smoothing, w=weights, per=False) ## NOTE: no need for cyclic as first and last point match
    # print(tck[0])
    return tck
  
# ----------------------------
# Main processing
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert to splines')
    parser.add_argument('-i','--input',help='Input SVG with polylines', required=True)
    parser.add_argument('-o','--output',help='Output SVG with polylines', required=True)
    parser.add_argument('-s','--smoothing', type=float, help='Smoothing, lower is more accurate but produces more control points', required=True)
    parser.add_argument('-l','--extremity_len', type=int, help='Length of extremity for open paths', required=False, default=4)
    parser.add_argument('-w','--extremity_w', type=float, help='Weight of extremity for open paths (lower means straighter)', required=False, default=0.1)
    args = vars(parser.parse_args())
    print(args)
    # Load polylines from SVG
    polylines = load_svg_polylines(args['input'])
    print("input: ",len(polylines))

    # To display the result (debug)
    # all_splines = [polyline_to_cubic_spline(poly, smoothing=1.0, num_points=50) for poly in polylines]
    # plt.show()

    paths = []
    for i in range(0,len(polylines)):
        poly = polylines[i]
        # print(poly)
        if len(poly) > 3:
            tck = polyline_to_tck_cubic_spline(poly, smoothing=args['smoothing'], cyclic=(poly[0] == poly[-1]),extremity_len=args['extremity_len'],extremity_w=args['extremity_w'])
            paths.append(bspline_to_svg_path(tck,cyclic=(poly[0] == poly[-1])))
        else:
            paths.append(polyline_to_svg_path(poly))
            
    print("output: ",len(paths))

    svg_content = wrap_svg(paths)
    with open(args['output'], "w") as f:
        f.write(svg_content)
    
    