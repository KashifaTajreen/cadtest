# cadhelp_app.py
# CADhelp - Orthographic Projection Generator (Streamlit)
# Single-file app. Dependencies: streamlit, pillow, numpy
# Save as cadhelp_app.py and run: streamlit run cadhelp_app.py

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import math, io, base64, textwrap
import numpy as np
from typing import List, Tuple

st.set_page_config(page_title="CADhelp", layout="wide")

# -----------------------
# Styling constants
# -----------------------
CANVAS_W = 1400
CANVAS_H = 900
MARGIN = 32
BG = (0, 0, 0)  # black background
WHITE = (255, 255, 255)
GRAY = (140, 140, 140)

# line widths
OUTLINE_W_FRONT = 6   # thick for front/top outlines
OUTLINE_W_TOP = 5
THIN_W = 1

# Boxes (front above XY, top below XY)
XY_Y = CANVAS_H // 2
FRONT_BOX = (MARGIN, MARGIN, CANVAS_W//2 - MARGIN//2, XY_Y - 18)
TOP_BOX   = (MARGIN, XY_Y + 18, CANVAS_W//2 - MARGIN//2, CANVAS_H - MARGIN)
INFO_BOX  = (CANVAS_W//2 + 10, MARGIN, CANVAS_W - MARGIN, CANVAS_H - MARGIN)

# fonts
def load_font(sz=14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", sz)
    except:
        return ImageFont.load_default()

# -----------------------
# Helper drawing functions
# -----------------------
def new_canvas():
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(img)
    return img, draw

def draw_layout(draw):
    # XY line
    draw.line([(MARGIN, XY_Y), (CANVAS_W - MARGIN, XY_Y)], fill=WHITE, width=1)
    # boxes
    draw.rectangle(FRONT_BOX, outline=WHITE, width=1)
    draw.rectangle(TOP_BOX, outline=WHITE, width=1)
    # labels
    f = load_font(16)
    draw.text((FRONT_BOX[0]+6, FRONT_BOX[1]-22), "Front view (above XY) — First Angle", font=f, fill=WHITE)
    draw.text((TOP_BOX[0]+6, TOP_BOX[3]+6), "Top view (below XY)", font=f, fill=WHITE)
    # info box
    draw.rectangle(INFO_BOX, outline=WHITE, width=1)

def mm_to_px_scale(max_mm, box_width_px):
    if max_mm <= 0: max_mm = 100.0
    return (box_width_px * 0.75) / max_mm

def in_box_transform(x_mm, y_mm, box, scale):
    # x to right, y upward
    left, top, right, bottom = box
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    px = cx + x_mm * scale
    py = cy - y_mm * scale
    return (px, py)

def rotate(pt, angle_deg):
    r = math.radians(angle_deg)
    x,y = pt
    return (x*math.cos(r) - y*math.sin(r), x*math.sin(r) + y*math.cos(r))

# -----------------------
# Geometry utilities
# -----------------------
def regular_polygon(n, side):
    # returns points centred at origin with given side length (approx)
    R = side / (2 * math.sin(math.pi / n))
    pts=[]
    for i in range(n):
        theta = 2*math.pi*i/n
        pts.append((R*math.cos(theta), R*math.sin(theta)))
    return pts

def circle_points(r, n=64):
    return [(r*math.cos(2*math.pi*i/n), r*math.sin(2*math.pi*i/n)) for i in range(n)]

# -----------------------
# Parsers (simple heuristics)
# -----------------------
import re
def extract_numbers(text):
    # returns floats found with optional units
    nums = re.findall(r'([-+]?\d*\.?\d+)\s*(?:mm|MM|cm|m| )?', text)
    return [float(x) for x in nums]

def extract_angles(text):
    angs = re.findall(r'([-+]?\d*\.?\d+)\s*(?:°|deg|degrees)?', text)
    return [float(x) for x in angs]

def parse_question(text):
    t = (text or "").lower()
    nums = extract_numbers(text)
    parsed={'type':None,'shape':None,'params':{},'raw':text}
    # point
    if 'point' in t or ('infront' in t and 'above' in t):
        parsed['type']='point'
        if nums:
            parsed['params']['infront']=nums[0]
            if len(nums)>1: parsed['params']['above']=nums[1]
        return parsed
    # line
    if 'line' in t or 'true length' in t or 'inclined' in t:
        parsed['type']='line'
        if nums:
            parsed['params']['true_length']=nums[0]
        # detect angles with keywords
        m = re.findall(r'(\d+)\s*(?:°|deg)', text)
        if m:
            # heuristics: first angle -> HP, second -> VP
            if len(m)>0: parsed['params']['angle_hp']=float(m[0])
            if len(m)>1: parsed['params']['angle_vp']=float(m[1])
        return parsed
    # lamina shapes
    if any(k in t for k in ['triangle','square','rectangular','rectangle','circle','pentagon','hexagon']):
        parsed['type']='lamina'
        for s in ['triangle','square','rectangular','rectangle','circle','pentagon','hexagon']:
            if s in t:
                parsed['shape']= 'rectangle' if s=='rectangular' else s
                break
        if nums:
            parsed['params']['nums']=nums
        # angle to hp maybe present
        a = re.search(r'makes\s+(\d+)\s*(?:°|deg)\s*to\s*hp', t)
        if a:
            parsed['params']['surface_angle']=float(a.group(1))
        return parsed
    # solids
    solids = ['prism','pyramid','cylinder','cone','cube','cuboid','hexagonal','pentagonal','triangular']
    if any(k in t for k in solids) or 'develop' in t or 'development' in t:
        # pick a specific type
        if 'cylinder' in t: parsed['type']='cylinder'
        elif 'cone' in t: parsed['type']='cone'
        elif 'square prism' in t or ('prism' in t and 'square' in t): parsed['type']='prism'; parsed['params']['base_sides']=4
        elif 'hexagonal prism' in t or 'hexagon' in t and 'prism' in t: parsed['type']='prism'; parsed['params']['base_sides']=6
        elif 'pentagonal prism' in t or 'pentagon' in t and 'prism' in t: parsed['type']='prism'; parsed['params']['base_sides']=5
        elif 'triangular prism' in t: parsed['type']='prism'; parsed['params']['base_sides']=3
        elif 'pyramid' in t:
            parsed['type']='pyramid'
            if 'hexagon' in t: parsed['params']['base_sides']=6
            elif 'pentagon' in t: parsed['params']['base_sides']=5
            elif 'square' in t: parsed['params']['base_sides']=4
            elif 'triangle' in t: parsed['params']['base_sides']=3
        elif 'cube' in t or 'cuboid' in t or 'rectangular prism' in t:
            parsed['type']='cuboid'
        else:
            parsed['type']='prism'
        if nums:
            parsed['params']['nums']=nums
        if 'develop' in t or 'development' in t or 'lateral' in t:
            parsed['params']['develop']=True
        return parsed

    # fallback: attempt line parse
    parsed['type']='line'
    if nums:
        parsed['params']['true_length']=nums[0]
    return parsed

# -----------------------
# Drawing primitives for feature types
# -----------------------
def draw_point_proj(draw, params):
    infront = float(params.get('infront', 30.0))
    above = float(params.get('above', 30.0))
    max_mm = max(infront, above, 100.0)
    scale = mm_to_px_scale(max_mm, min(FRONT_BOX[2]-FRONT_BOX[0], FRONT_BOX[3]-FRONT_BOX[1]))
    # front point (x=0,z=above)
    p_front = in_box_transform(0, above, FRONT_BOX, scale)
    p_top   = in_box_transform(0, infront, TOP_BOX, scale)
    draw.ellipse([p_front[0]-6,p_front[1]-6,p_front[0]+6,p_front[1]+6], outline=WHITE, width=OUTLINE_W_FRONT)
    draw.ellipse([p_top[0]-5,p_top[1]-5,p_top[0]+5,p_top[1]+5], outline=WHITE, width=OUTLINE_W_TOP)
    draw.line([p_front, p_top], fill=WHITE, width=THIN_W)
    return [f"Point: {infront} mm infront of VP, {above} mm above HP"]

def draw_line_proj(draw, params):
    # We construct endpoints in 3D consistent with given true length L and inclinations to HP (alpha) and VP (beta)
    L = float(params.get('true_length', 80.0))
    alpha = float(params.get('angle_hp', 30.0))  # inclination to HP
    beta = float(params.get('angle_vp', 45.0))   # inclination to VP
    # true geometry: line vector components:
    # component normal to HP (z) = L*sin(alpha)
    z_comp = L * math.sin(math.radians(alpha))
    # component projected on HP plane (plan length)
    plan = L * math.cos(math.radians(alpha))
    # angle in plan relative to VP axis determined by beta: plan's tilt such that apparent front length = L*cos(beta)
    # approximate approach: choose plan orientation so top view length = plan and front apparent matches
    # choose plan angle phi such that apparent front = plan * cos(phi) = L * cos(beta) => cos(phi) = cos(beta) * L / plan
    # to avoid division by zero:
    if plan == 0:
        phi = 0
    else:
        val = (L * math.cos(math.radians(beta))) / plan
        val = max(-1.0, min(1.0, val))
        phi = math.degrees(math.acos(val))
    # choose a start point A in plan (ax, ay) (mm)
    Ax, Ay, Az = -L*0.15, -10.0, 10.0
    Bx = Ax + plan * math.cos(math.radians(phi))
    By = Ay + plan * math.sin(math.radians(phi))
    Bz = Az + z_comp
    # convert to drawing
    max_mm = max(L, abs(Ax), abs(Bx), abs(Ay), abs(By), abs(Az), abs(Bz), 120.0)
    scale = mm_to_px_scale(max_mm, FRONT_BOX[2]-FRONT_BOX[0])
    A_front = in_box_transform(Ax, Az, FRONT_BOX, scale)
    B_front = in_box_transform(Bx, Bz, FRONT_BOX, scale)
    A_top = in_box_transform(Ax, Ay, TOP_BOX, scale)
    B_top = in_box_transform(Bx, By, TOP_BOX, scale)
    # draw
    draw.line([A_front, B_front], fill=WHITE, width=OUTLINE_W_FRONT)
    draw.line([A_top, B_top], fill=WHITE, width=OUTLINE_W_TOP)
    draw.ellipse([A_front[0]-6,A_front[1]-6,A_front[0]+6,A_front[1]+6], outline=WHITE, width=OUTLINE_W_FRONT)
    draw.ellipse([B_front[0]-6,B_front[1]-6,B_front[0]+6,B_front[1]+6], outline=WHITE, width=OUTLINE_W_FRONT)
    draw.ellipse([A_top[0]-5,A_top[1]-5,A_top[0]+5,A_top[1]+5], outline=WHITE, width=OUTLINE_W_TOP)
    draw.ellipse([B_top[0]-5,B_top[1]-5,B_top[0]+5,B_top[1]+5], outline=WHITE, width=OUTLINE_W_TOP)
    # projectors
    draw.line([A_front, A_top], fill=WHITE, width=THIN_W)
    draw.line([B_front, B_top], fill=WHITE, width=THIN_W)
    # annotations
    apparent_top = math.hypot(Bx-Ax, By-Ay)
    apparent_front = math.hypot(B_front[0]-A_front[0], B_front[1]-A_front[1]) / scale if scale!=0 else 0
    info = [
        f"True length = {L:.2f} mm",
        f"Inclination to HP (alpha) = {alpha:.2f}°",
        f"Inclination to VP (beta) = {beta:.2f}°",
        f"Top (plan) apparent length ≈ {apparent_top:.2f} mm",
        f"Front apparent (pixels->mm) ≈ {apparent_front:.2f} mm"
    ]
    return info

def draw_lamina_proj(draw, parsed):
    shape = parsed.get('shape','triangle')
    nums = parsed.get('params', {}).get('nums', [])
    surface_angle = parsed.get('params', {}).get('surface_angle', 60.0)
    edge_rot = parsed.get('params', {}).get('edge_rot', 25.0)
    # determine size in mm
    if nums:
        size = float(nums[0])
    else:
        size = 40.0
    # compute scale based on largest dimension
    max_mm = max(size, 100.0)
    scale = mm_to_px_scale(max_mm, FRONT_BOX[2]-FRONT_BOX[0])
    cx_f = (FRONT_BOX[0]+FRONT_BOX[2])/2
    cy_f = (FRONT_BOX[1]+FRONT_BOX[3])/2
    cx_t = (TOP_BOX[0]+TOP_BOX[2])/2
    cy_t = (TOP_BOX[1]+TOP_BOX[3])/2
    info=[]
    # FRONT view: shape foreshortened by cos(surface_angle)
    foreshorten = abs(math.cos(math.radians(surface_angle)))
    if shape in ('triangle',):
        # equilateral triangle
        h_true = math.sqrt(3)/2 * size
        # front triangle
        A=(cx_f, cy_f - h_true/2 * scale * foreshorten)
        B=(cx_f - (size/2)*scale, cy_f + h_true/2 * scale * foreshorten)
        C=(cx_f + (size/2)*scale, cy_f + h_true/2 * scale * foreshorten)
        draw.line([A,B,C,A], fill=WHITE, width=OUTLINE_W_FRONT)
        # top view: rotate by edge_rot, true size
        pts_top=[]
        for px,py in [(-size/2,-h_true/3),(size/2,-h_true/3),(0,2*h_true/3)]:
            rx,ry = rotate((px,py), edge_rot)
            pts_top.append((cx_t + rx*scale, cy_t + ry*scale))
        draw.line([pts_top[0],pts_top[1],pts_top[2],pts_top[0]], fill=WHITE, width=OUTLINE_W_TOP)
        for p,q in zip([A,B,C], pts_top):
            draw.line([p,q], fill=WHITE, width=THIN_W)
        info.append(f"Equilateral triangle side {size} mm; surface to HP {surface_angle}°")
    elif shape in ('square','rectangle'):
        if shape=='square':
            w = size
            h = size
        else:
            if len(nums)>=2:
                w = float(nums[0]); h = float(nums[1])
            else:
                w = size; h = size*1.6
        hw = (w/2)*scale
        hh = (h/2)*scale * foreshorten
        left = cx_f - hw; right = cx_f + hw; top = cy_f - hh; bottom = cy_f + hh
        draw.rectangle([left,top,right,bottom], outline=WHITE, width=OUTLINE_W_FRONT)
        # top view rotated
        corners=[(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]
        pts_top=[]
        for px,py in corners:
            rx,ry = rotate((px,py), edge_rot)
            pts_top.append((cx_t + rx*scale, cy_t + ry*scale))
        draw.line(pts_top + [pts_top[0]], fill=WHITE, width=OUTLINE_W_TOP)
        for i,(p,q) in enumerate(zip([(left,top),(right,top),(right,bottom),(left,bottom)], pts_top)):
            draw.line([p,q], fill=WHITE, width=THIN_W)
        info.append(f"{shape.title()} {w} x {h} mm; surface to HP {surface_angle}°")
    elif shape=='circle':
        diam = size
        r = (diam/2)*scale
        rx = r; ry = r * foreshorten
        draw.ellipse([cx_f-rx, cy_f-ry, cx_f+rx, cy_f+ry], outline=WHITE, width=OUTLINE_W_FRONT)
        draw.ellipse([cx_t-r, cy_t-r, cx_t+r, cy_t+r], outline=WHITE, width=OUTLINE_W_TOP)
        draw.line([(cx_f-rx,cy_f),(cx_t-r,cy_t)], fill=WHITE, width=THIN_W)
        info.append(f"Circle diameter {diam} mm; surface to HP {surface_angle}°")
    elif shape in ('pentagon','hexagon'):
        n = 5 if shape=='pentagon' else 6
        side = size
        pts_reg = regular_polygon(n, side)
        # front: foreshortened vertical
        pts_front = [(cx_f + x*scale, cy_f + y*scale * foreshorten) for (x,y) in pts_reg]
        pts_top = [(cx_t + rotate((x,y), edge_rot)[0]*scale, cy_t + rotate((x,y), edge_rot)[1]*scale) for (x,y) in pts_reg]
        draw.line(pts_front + [pts_front[0]], fill=WHITE, width=OUTLINE_W_FRONT)
        draw.line(pts_top + [pts_top[0]], fill=WHITE, width=OUTLINE_W_TOP)
        for pf,pt in zip(pts_front, pts_top):
            draw.line([pf, pt], fill=WHITE, width=THIN_W)
        info.append(f"Regular {n}-gon side ~{side} mm; surface to HP {surface_angle}°")
    else:
        info.append("Unknown lamina shape")
    return info

# -----------------------
# Solids & development drawing
# -----------------------
def draw_prism_proj(draw, params):
    base_sides = int(params.get('base_sides', 4))
    nums = params.get('nums', [])
    if nums:
        if len(nums) >= 3:
            base_w = float(nums[0]); base_d = float(nums[1]); height = float(nums[2])
        elif len(nums) == 2:
            base_w = float(nums[0]); base_d = float(nums[1]); height = 80.0
        else:
            base_w = float(nums[0]); base_d = base_w*0.8; height = 80.0
    else:
        base_w, base_d, height = 50.0, 40.0, 80.0
    # scale
    max_mm = max(base_w, base_d, height, 120.0)
    scale = mm_to_px_scale(max_mm, FRONT_BOX[2]-FRONT_BOX[0])
    cx_f = (FRONT_BOX[0]+FRONT_BOX[2])/2
    cy_f = (FRONT_BOX[1]+FRONT_BOX[3])/2
    cx_t = (TOP_BOX[0]+TOP_BOX[2])/2
    cy_t = (TOP_BOX[1]+TOP_BOX[3])/2
    # front view: rectangle width=base_w, height=height
    left = cx_f - (base_w/2)*scale; right = cx_f + (base_w/2)*scale
    top = cy_f - (height/2)*scale; bottom = cy_f + (height/2)*scale
    draw.rectangle([left,top,right,bottom], outline=WHITE, width=OUTLINE_W_FRONT)
    # top view: base_w x base_d rectangle
    tlx = cx_t - (base_w/2)*scale; tly = cy_t - (base_d/2)*scale
    brx = cx_t + (base_w/2)*scale; bry = cy_t + (base_d/2)*scale
    draw.rectangle([tlx,tly,brx,bry], outline=WHITE, width=OUTLINE_W_TOP)
    # projectors corners
    corners_front = [(left,top),(right,top),(right,bottom),(left,bottom)]
    corners_top = [(tlx,tly),(brx,tly),(brx,bry),(tlx,bry)]
    for a,b in zip(corners_front, corners_top):
        draw.line([a,b], fill=WHITE, width=THIN_W)
    info=[f"Prism: base {base_w} x {base_d} mm, height {height} mm"]
    # development (parallel-line) - show as strip on right within INFO_BOX
    dev_left = INFO_BOX[0] + 12
    dev_top = INFO_BOX[1] + 20
    face_w = (base_w/ base_w) * (base_w*scale)  # face width in px
    face_h = height * scale
    # draw n faces horizontally
    for i in range(base_sides):
        x0 = dev_left + i*(face_w + 6)
        draw.rectangle([x0, dev_top, x0+face_w, dev_top+face_h], outline=WHITE, width=OUTLINE_W_FRONT)
        # fold lines
        if i < base_sides-1:
            draw.line([x0+face_w, dev_top, x0+face_w, dev_top+face_h], fill=WHITE, width=THIN_W)
    # draw base positions above
    return info

def draw_pyramid_proj(draw, params):
    base_sides = int(params.get('base_sides', 4))
    nums = params.get('nums', [])
    if nums:
        base_side = float(nums[0])
        height = float(nums[1]) if len(nums)>1 else 80.0
    else:
        base_side = 50.0; height = 80.0
    max_mm = max(base_side, height, 120.0)
    scale = mm_to_px_scale(max_mm, FRONT_BOX[2]-FRONT_BOX[0])
    cx_f = (FRONT_BOX[0]+FRONT_BOX[2])/2
    cy_f = (FRONT_BOX[1]+FRONT_BOX[3])/2
    cx_t = (TOP_BOX[0]+TOP_BOX[2])/2
    cy_t = (TOP_BOX[1]+TOP_BOX[3])/2
    # front: isosceles triangle with apex at top and base = base_side
    left = cx_f - (base_side/2)*scale
    right = cx_f + (base_side/2)*scale
    base_y = cy_f + (height/2)*scale
    apex = (cx_f, cy_f - (height/2)*scale)
    A=(left, base_y); B=(right, base_y)
    draw.line([A,B,apex,A], fill=WHITE, width=OUTLINE_W_FRONT)
    # top: regular polygon base
    pts = regular_polygon(base_sides, base_side)
    pts_top = [(cx_t + x*scale, cy_t + y*scale) for x,y in pts]
    draw.line(pts_top + [pts_top[0]], fill=WHITE, width=OUTLINE_W_TOP)
    # project base vertices to front base approximate
    for i,pt in enumerate(pts_top[:min(3,len(pts_top))]):
        # connect first three to indicate projectors
        draw.line([pt, (left + i*(right-left)/max(1,len(pts_top)-1), base_y)], fill=WHITE, width=THIN_W)
    info=[f"Pyramid base side ~{base_side} mm, sides={base_sides}, height {height} mm"]
    # development - triangular lateral faces in INFO_BOX
    dev_left = INFO_BOX[0] + 12
    dev_top = INFO_BOX[1] + 20
    for i in range(base_sides):
        # approximate lateral face as isosceles triangle
        b = base_side*scale
        slant = math.hypot((base_side/2), height) * scale
        x0 = dev_left + i*(b + 6)
        # triangle points
        draw.line([(x0, dev_top + slant), (x0 + b/2, dev_top), (x0 + b, dev_top + slant)], fill=WHITE, width=OUTLINE_W_FRONT)
    return info

def draw_cylinder_proj(draw, params):
    diam = float(params.get('nums',[40.0])[0]) if params.get('nums') else float(params.get('diameter',40.0))
    height = float(params.get('height', 80.0))
    max_mm = max(diam, height, 120.0)
    scale = mm_to_px_scale(max_mm, FRONT_BOX[2]-FRONT_BOX[0])
    r = (diam/2)*scale
    cx_f = (FRONT_BOX[0]+FRONT_BOX[2])/2
    cy_f = (FRONT_BOX[1]+FRONT_BOX[3])/2
    cx_t = (TOP_BOX[0]+TOP_BOX[2])/2
    cy_t = (TOP_BOX[1]+TOP_BOX[3])/2
    left = cx_f - r; right = cx_f + r
    top = cy_f - (height/2)*scale; bottom = cy_f + (height/2)*scale
    # front: rectangle with elliptical top/bottom (simplified: rectangle)
    draw.rectangle([left, top, right, bottom], outline=WHITE, width=OUTLINE_W_FRONT)
    # top: circle true size
    draw.ellipse([cx_t-r, cy_t-r, cx_t+r, cy_t+r], outline=WHITE, width=OUTLINE_W_TOP)
    # projectors
    draw.line([(left, top), (cx_t-r, cy_t-r)], fill=WHITE, width=THIN_W)
    draw.line([(right, top), (cx_t+r, cy_t-r)], fill=WHITE, width=THIN_W)
    info=[f"Cylinder diameter {diam} mm, height {height} mm"]
    return info

def draw_cone_proj(draw, params):
    diam = float(params.get('nums',[40.0])[0]) if params.get('nums') else float(params.get('diameter',40.0))
    height = float(params.get('height', 80.0))
    max_mm = max(diam, height, 120.0)
    scale = mm_to_px_scale(max_mm, FRONT_BOX[2]-FRONT_BOX[0])
    r = (diam/2)*scale
    cx_f = (FRONT_BOX[0]+FRONT_BOX[2])/2
    cy_f = (FRONT_BOX[1]+FRONT_BOX[3])/2
    cx_t = (TOP_BOX[0]+TOP_BOX[2])/2
    cy_t = (TOP_BOX[1]+TOP_BOX[3])/2
    base_left = (cx_f - r, cy_f + (height/2)*scale)
    base_right = (cx_f + r, cy_f + (height/2)*scale)
    apex = (cx_f, cy_f - (height/2)*scale)
    draw.line([base_left, base_right], fill=WHITE, width=OUTLINE_W_FRONT)
    draw.line([base_left, apex, base_right], fill=WHITE, width=OUTLINE_W_FRONT)
    draw.ellipse([cx_t-r, cy_t-r, cx_t+r, cy_t+r], outline=WHITE, width=OUTLINE_W_TOP)
    draw.line([base_left, (cx_t-r, cy_t)], fill=WHITE, width=THIN_W)
    draw.line([base_right, (cx_t+r, cy_t)], fill=WHITE, width=THIN_W)
    info=[f"Cone base diameter {diam} mm, height {height} mm"]
    return info

def draw_cuboid_proj(draw, params):
    if params.get('nums'):
        nums=params.get('nums')
        if len(nums)>=3:
            w,d,h = float(nums[0]), float(nums[1]), float(nums[2])
        elif len(nums)==2:
            w,d,h=float(nums[0]),float(nums[1]),80.0
        else:
            w=50.0; d=40.0; h=50.0
    else:
        w=float(params.get('w',50.0)); d=float(params.get('d',40.0)); h=float(params.get('h',50.0))
    max_mm = max(w,d,h,120.0)
    scale = mm_to_px_scale(max_mm, FRONT_BOX[2]-FRONT_BOX[0])
    cx_f = (FRONT_BOX[0]+FRONT_BOX[2])/2
    cy_f = (FRONT_BOX[1]+FRONT_BOX[3])/2
    cx_t = (TOP_BOX[0]+TOP_BOX[2])/2
    cy_t = (TOP_BOX[1]+TOP_BOX[3])/2
    left = cx_f - (w/2)*scale; right = cx_f + (w/2)*scale
    top = cy_f - (h/2)*scale; bottom = cy_f + (h/2)*scale
    tlx = cx_t - (w/2)*scale; tly = cy_t - (d/2)*scale
    brx = cx_t + (w/2)*scale; bry = cy_t + (d/2)*scale
    draw.rectangle([left,top,right,bottom], outline=WHITE, width=OUTLINE_W_FRONT)
    draw.rectangle([tlx,tly,brx,bry], outline=WHITE, width=OUTLINE_W_TOP)
    for a,b in [((left,top),(tlx,tly)),((right,top),(brx,tly)),((right,bottom),(brx,bry)),((left,bottom),(tlx,bry))]:
        draw.line([a,b], fill=WHITE, width=THIN_W)
    info=[f"Cuboid {w} x {d} x {h} mm"]
    return info

# -----------------------
# Main App UI & logic
# -----------------------
st.title("CADhelp — Orthographic Projection Generator")
st.markdown("Type a question in the textbook style (examples in the images). The app parses the text and generates front & top orthographic projections. Use the panels to override numeric parameters if needed.")

# small logo + app name centered
logo = Image.new("RGB", (140,140), BG)
ld = ImageDraw.Draw(logo)
# simple engineering-style symbol: compass + square
ld.ellipse([20,20,120,120], outline=WHITE, width=3)
ld.line([70,30,70,110], fill=WHITE, width=3)  # axis
ld.line([30,70,110,70], fill=WHITE, width=3)
# display
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image(logo, width=80)
    st.markdown("<h2 style='color:white;text-align:center;background:black;padding:6px;border-radius:4px'>CADhelp</h2>", unsafe_allow_html=True)

# show optional uploaded photo for reference
uploaded = st.file_uploader("Upload reference photo (optional) — shows for your reference, OCR not performed", type=['png','jpg','jpeg'])
if uploaded:
    st.image(uploaded, caption="Reference photo (not parsed automatically)", use_column_width=False, width=320)

st.markdown("---")

# bottom input area for typed question (the user specifically asked to keep the space at bottom; we place input near bottom visually by layout)
# but in streamlit, we just provide a large text_area near the end of controls
st.sidebar.header("Controls (optional)")
force_type = st.sidebar.selectbox("Force type (optional)", ["Auto","Point","Line","Lamina","Prism","Pyramid","Cylinder","Cone","Cuboid","Develop"])
override_text = st.sidebar.text_input("Override params (comma separated) e.g. true_length=80, angle_hp=30, angle_vp=45, base_w=50, base_d=40, height=80")

# parse overrides
overrides={}
if override_text.strip():
    for piece in override_text.split(','):
        if '=' in piece:
            k,v = piece.split('=',1)
            try:
                overrides[k.strip()]=float(v.strip())
            except:
                overrides[k.strip()]=v.strip()

st.markdown("### Type the question (examples):\n- Line AB 80 mm long inclined 30° to HP and 45° to VP\n- Triangular lamina side 30 mm resting on one side on HP; surface makes 60° to HP and corner nearest to VP 30 mm in front of VP\n- Cylinder base diameter 80 mm height 100 mm cut by plane inclined 45° to HP\n\n(You can force type or override numbers in sidebar.)")

q_text = st.text_area("Question (type here):", height=130)

# parse
parsed = parse_question(q_text)
# apply forced type if set
if force_type != "Auto":
    parsed['type'] = force_type.lower()
# merge overrides into parsed params
parsed_params = parsed.get('params', {})
for k,v in overrides.items():
    parsed_params[k]=v
parsed['params']=parsed_params

st.markdown("**Parsed (heuristic)**")
st.json(parsed)

# Generate button
if st.button("Generate projection"):
    img, draw = new_canvas()
    draw_layout(draw)  # boxes and labels
    info_lines = []
    t = parsed.get('type')
    try:
        if t == 'point' or force_type=="Point":
            info_lines = draw_point_proj(draw, parsed.get('params', {}))
        elif t == 'line' or force_type=="Line":
            info_lines = draw_line_proj(draw, parsed.get('params', {}))
        elif t == 'lamina' or force_type=="Lamina":
            info_lines = draw_lamina_proj(draw, parsed)
        elif t == 'prism' or t=='prism' or force_type=="Prism":
            info_lines = draw_prism_proj(draw, parsed.get('params', {}))
        elif t == 'pyramid' or force_type=="Pyramid":
            info_lines = draw_pyramid_proj(draw, parsed.get('params', {}))
        elif t == 'cylinder' or force_type=="Cylinder":
            info_lines = draw_cylinder_proj(draw, parsed.get('params', {}))
        elif t == 'cone' or force_type=="Cone":
            info_lines = draw_cone_proj(draw, parsed.get('params', {}))
        elif t == 'cuboid' or force_type=="Cuboid":
            info_lines = draw_cuboid_proj(draw, parsed.get('params', {}))
        else:
            # fallback: draw a representative line
            info_lines = draw_line_proj(draw, parsed.get('params', {}))
    except Exception as e:
        st.error(f"Error while generating: {e}")
        info_lines = [f"Error: {e}"]

    # draw info lines in info box area
    left, top, right, bottom = INFO_BOX
    f = load_font(14)
    y = top + 12
    for line in info_lines:
        draw.text((left+8, y), str(line), font=f, fill=WHITE)
        y += 18

    # show image
    st.image(img, use_column_width=True)
    # download button
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b = buf.getvalue()
    b64 = base64.b64encode(b).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="cadhelp_projection.png">Download PNG</a>'
    st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.caption("Notes: This app uses geometric approximations to produce textbook-style first-angle and auxiliary-plane projections. For full DWG/AutoCAD compatibility you can export the PNG and trace or import into CAD. If you want full DXF/DWG output, I can extend the app to export DXF (requires ezdxf library).")

