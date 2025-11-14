import argparse
from enum import Enum

import jax
import numpy as np
from svgpathtools import Line, Path, paths2svg, svg2paths

import jax.numpy as jnp

import cglib.backend
import cglib.cycle
import cglib.fdm_aa
import cglib.polyline

import cglib.transform as transform


class DataToExport(Enum):
    """
    Enumeration of the data that can be exported to svg.

    Attributes
    ----------
    CYCLE:
        The cycle after the stitching.
    CYCLES:
        The cycles after the contouring and before the stitching.
    """
    CYCLE = 'cycle'
    CYCLES = 'cycles'

import math
from svgpathtools import Line, Path

def smooth_path(path_i,k):

    pts = [path_i[0].start] + [seg.end for seg in path_i]

    for _ in range(0,k):
      smooth = [0] * len(pts)
      for i in range(0, len(pts)):
          im1 = (i-1 + len(pts)) % len(pts)
          ip1 = (i+1) % len(pts)
          lm1 = abs(pts[im1] - pts[i])
          lp1 = abs(pts[ip1] - pts[i])
          smooth[i] = (pts[im1]*lm1 + pts[i]*(lm1+lp1) + pts[ip1]*lp1) / (2*lm1+2*lp1)
      pts = smooth

    path = Path()
    for i in range(1, len(pts)):
      path.append(Line(pts[i-1],pts[i]))
    path.append(Line(pts[len(pts)-1],pts[0]))
    return path

def cut_polyline_on_curvature(path_i, radius_threshold=5.0, k_remove=0):
    """
    Cuts a polyline (array of Line objects) wherever the local curvature exceeds `curvature_threshold`.
    Optionally removes `k_remove` segments before and after.
    Returns list of Path objects (compatible with wsvg).
    """

    # Convert Line objects to complex points
    pts = [path_i[0].start] + [seg.end for seg in path_i]
    cyclic = pts[0] == pts[-1]

    def radius(p_prev, p, p_next):
      """Compute curvature as 1 / circumcircle radius through three points."""
      a = abs(p - p_prev)
      b = abs(p_next - p)
      c = abs(p_next - p_prev)
      # Heron's formula for area of triangle
      s    = (a + b + c) / 2
      sqarea = abs(s * (s - a) * (s - b) * (s - c))
      if sqarea == 0:
          return 0
      area = math.sqrt(sqarea)
      R = (a * b * c) / (4 * area)
      return R

    # Compute local curvature at interior points
    radii = [0] * len(pts)
    for i in range(0, len(pts)-1):
      im1 = (i-1 + len(pts)-1) % (len(pts)-1)
      ip1 = (i+1) % (len(pts)-1)
      radii[i] = radius(pts[im1], pts[i], pts[ip1])

    # Detect high-curvature regions
    cut_indices = [i for i, k in enumerate(radii) if k < radius_threshold]

    # Mark segments to remove around those regions
    to_remove = set()
    for idx in cut_indices:
        for j in range(idx - k_remove, idx + k_remove + 1):
            if 0 <= j < len(path_i):
                to_remove.add(j)

    # Build resulting Paths
    subpaths = []
    current = []
    for i, seg in enumerate(path_i):
        if i in to_remove:
            if current:
                subpaths.append(Path(*current))
                current = []
        else:
            current.append(seg)
    if current:
        subpaths.append(Path(*current))
    # check if first/last should be joined due to cycle
    if cyclic and len(subpaths) > 1:
        if subpaths[0][0].start == subpaths[-1][-1].end:            
            # join them 
            for l in subpaths[0]:
                subpaths[-1].append(l)
            subpaths.pop(0)
    
    return subpaths

def cycles_to_svg(input_param_filename, svg_out_filename, split_radius=4.0, smoothing=10):

    device_cpu, _ = cglib.backend.get_cpu_and_gpu_devices()

    parameters = cglib.fdm_aa.Parameters()
    parameters.load(input_param_filename)

    # Load svg paths to get svg attributes
    _, _, svg_attributes = svg2paths(
        parameters.svg_path, return_svg_attributes=True)

    # The shape domain size is determined by the SVG width and heigh
    # -2: remove the unit
    svg_width = float(svg_attributes['width'][:-2])
    svg_height = float(svg_attributes['height'][:-2])
    # [x, y]
    shape_domain_size = jnp.array([svg_width, svg_height])

    trans = transform.translate(jnp.array([0., -shape_domain_size[1]*0.5]))
    scale = transform.scale(jnp.array([1., -1.]))
    upside_down_to_right_side_up = jnp.linalg.inv(trans) @ scale @ trans
    upside_down_to_right_side_up = jax.device_put(upside_down_to_right_side_up, device_cpu)

    cycles = cglib.cycle.load(parameters.cycles_filename)
    cycles = jax.device_put(cycles, device_cpu)
    print(f"Cycle count before stitching: {cycles.cycle_count}")
    polylines = cglib.cycle.to_polyline_full_nan(cycles)
    polylines = jax.jit(cglib.cycle.to_polyline)(cycles, polylines)
    polylines: cglib.polyline.Polyline = jax.device_get(polylines)
    paths = []

    for i in range(cycles.cycle_count):
        polylines_point_i = jax.jit(jax.vmap(transform.apply_to_point, (None, 0)))(
        upside_down_to_right_side_up, polylines.point[i])
        cycle_polyline_2dpoint = np.array(polylines_point_i)
        point_count_i = int(polylines.data[i, 1])
        cycle_polyline_2dpoint = cycle_polyline_2dpoint[:point_count_i, :]

        path_i = Path()

        for j in range(1, point_count_i):
            path_i.append(Line(
                complex(cycle_polyline_2dpoint[j-1, 0],
                        cycle_polyline_2dpoint[j-1, 1]),
                complex(cycle_polyline_2dpoint[j, 0], cycle_polyline_2dpoint[j, 1])))
        # Add last point
        path_i.append(Line(
            complex(cycle_polyline_2dpoint[point_count_i-1, 0],
                    cycle_polyline_2dpoint[point_count_i-1, 1]),
            complex(cycle_polyline_2dpoint[0, 0], cycle_polyline_2dpoint[0, 1])))

        # Cut paths based on curvature
        path_i = smooth_path(path_i,smoothing)
        #                           ^ smoothing strength (iterations)
        if False:
          paths.append(path_i)
        else:
          paths_i = cut_polyline_on_curvature(path_i,split_radius,0)
          #                                          ^ allowed min radius (cuts below)
          for p in paths_i:
            paths.append(p)
    paths2svg.wsvg(paths, filename=svg_out_filename,
                   svg_attributes=svg_attributes)

def cycle_to_svg(input_param_filename, svg_out_filename, split_radius=4.0, smoothing=10):

    device_cpu, _ = cglib.backend.get_cpu_and_gpu_devices()

    parameters = cglib.fdm_aa.Parameters()
    parameters.load(input_param_filename)
    cycle_polyline = cglib.polyline.load(parameters.cycle_polyline_filename)
    cycle_polyline_2dpoint: np.ndarray = cycle_polyline.point[0]
    cycle_polyline_2dpoint = jax.device_put(cycle_polyline_2dpoint, device_cpu)

    # Load svg paths to get svg attributes
    _, _, svg_attributes = svg2paths(
        parameters.svg_path, return_svg_attributes=True)

    # The shape domain size is determined by the SVG width and heigh
    # -2: remove the unit
    svg_width = float(svg_attributes['width'][:-2])
    svg_height = float(svg_attributes['height'][:-2])
    # [x, y]
    shape_domain_size = jnp.array([svg_width, svg_height])

    trans = transform.translate(jnp.array([0., -shape_domain_size[1]*0.5]))
    scale = transform.scale(jnp.array([1., -1.]))
    upside_down_to_right_side_up = jnp.linalg.inv(trans) @ scale @ trans
    upside_down_to_right_side_up = jax.device_put(upside_down_to_right_side_up, device_cpu)

    path_i = Path()

    cycle_polyline_2dpoint = jax.jit(jax.vmap(transform.apply_to_point, (None, 0)))(
        upside_down_to_right_side_up, cycle_polyline_2dpoint)

    cycle_polyline_2dpoint = jax.device_get(cycle_polyline_2dpoint)
    cycle_polyline_2dpoint = np.array(cycle_polyline_2dpoint)

    for j in range(1, cycle_polyline_2dpoint.shape[0]):
        path_i.append(Line(
            complex(cycle_polyline_2dpoint[j-1, 0],
                    cycle_polyline_2dpoint[j-1, 1]),
            complex(cycle_polyline_2dpoint[j, 0], cycle_polyline_2dpoint[j, 1])))
    # Add last point
    path_i.append(Line(
        complex(cycle_polyline_2dpoint[cycle_polyline_2dpoint.shape[0]-1, 0],
                cycle_polyline_2dpoint[cycle_polyline_2dpoint.shape[0]-1, 1]),
        complex(cycle_polyline_2dpoint[0, 0], cycle_polyline_2dpoint[0, 1])))

    paths2svg.wsvg(path_i, filename=svg_out_filename,
                   svg_attributes=svg_attributes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert the trajectory to SVG. Input: The JSON file used by fill_2d_shape.py to genrate the trajectory and a path for the SVG. Output: The trajectory in SVG format.'
    )
    parser.add_argument(
        "input_filename", help="The JSON file used by fill_2d_shape.py to genrate the trajectory.")
    parser.add_argument(
        "datatoexport", help="The data to export. Either `cycle` (the cycle after stitching) or `cycles` (the cycles before stitching).")
    parser.add_argument("svg_out_filename", help="The path of the output SVG.")
    parser.add_argument("--split_radius", help="Split radius (4.0).", type=float, default=4.0)
    parser.add_argument("--smoothing", help="Smoothing strength (10).", type=int, default=10)
    args = parser.parse_args()
    print(args)
    
    input_param_filename = args.input_filename
    datatoexport = args.datatoexport
    svg_out_filename = args.svg_out_filename

    if datatoexport == DataToExport.CYCLES.value:
        cycles_to_svg(input_param_filename, svg_out_filename, args.split_radius, args.smoothing)
    elif datatoexport == DataToExport.CYCLE.value:
        cycle_to_svg(input_param_filename, svg_out_filename)
    else:
        exit("Either `cycle` or `cycles` is accepted for the second positional argument.")
