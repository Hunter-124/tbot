"""trig_plug: guard-opening locating plug with a jack screw (rev AG).

Gun frame (design/layout.md): +X to muzzle, +Y gun-left, +Z up, origin at the
trigger finger-face contact at rest. Run:
    python3 design/cad_src/trig_plug.py <guard_measured.stl> <out.step>

The plug sits in the front of the trigger-guard opening, between the right
clamp land and the servo-bracket land (the taper planes |Y| = h(X) + 0.85),
on the existing U18 and U28 through-bolts. An M3 x 18 jack screw on a
horizontal axis at Z = JACK_Z pushes the front inside wall of the guard
forward, which pulls the whole trigger set back onto the guard cradle: the
front leg of the guard is then clamped between the jack tip (inside) and the
cradle (outside) instead of relying on friction on the tapered sides.
"""
import sys
from build123d import (Box, Cylinder, Plane, Pos, Rot, Location, Axis, import_step,
                       export_step, extrude, offset, Kind, Polyline, make_face,
                       RegularPolygon, Solid, Compound, Vector)

# ---- parameters -------------------------------------------------------------
X_REAR = 15.5          # rear face; servo_lever max X at rest is 13.56
Z_TOP = 8.5            # top of plug (opening top/receiver is Z 16); 1.8 over the counterbore
X_FRONT = 33.5         # flat jack face; model wall at Z 3.5 is X 36.45
GAP = 0.7              # clearance to the modelled guard inner surface
LAND_A, LAND_B = 4.493 - 0.05, 0.08421   # |Y| = A + B*X : clamp/bracket taper lands, 0.05 inside
U18 = (18.5, -7.75)    # through-bolt axes (X, Z), along Y
U28 = (28.0, -3.0)
BOLT_D = 3.4
JACK_Z = 3.5           # jack axis, parallel to X, at Y 0
CB_D, CB_END = 6.4, 24.5     # head counterbore from the rear face
TAP_D, TAP_END = 2.8, 27.0   # thread-forming section (prevailing-torque lock)
NUT_X0, NUT_T = 27.3, 2.7    # trapped M3 nut pocket (AF 5.5 -> 5.75 slot)
NUT_AF = 5.75
CLR_D = 3.4                  # nut to front face

def opening_profile(guard_stl):
    """Opening of the guard at Y 0 in (X, Z), shrunk by GAP (shapely polygon)."""
    import trimesh
    from shapely.geometry import Polygon, box, Point
    from shapely.ops import unary_union
    g = trimesh.load(guard_stl)
    sec = g.section(plane_origin=[0, 0, 0], plane_normal=[0, 1, 0])
    loops = [Polygon([(p[0], p[2]) for p in e]).buffer(0) for e in sec.discrete]
    region = box(-12, -15, 45, 15.99).difference(unary_union(loops))
    if region.geom_type != 'Polygon':
        region = [r for r in region.geoms if r.contains(Point(20, 0))][0]
    return region.buffer(-GAP, join_style=1)

def build(guard_stl):
    from shapely.geometry import box
    allowed = opening_profile(guard_stl).intersection(box(X_REAR, -20, X_FRONT, Z_TOP))
    allowed = allowed.simplify(0.02)
    pts = [(x, z) for x, z in list(allowed.exterior.coords)[:-1]]
    prof = make_face(Polyline(*[(x, 0, z) for x, z in pts], close=True))
    blk = extrude(prof, amount=15, both=True, dir=(0, 1, 0))
    # raw block, then trim to the two taper lands (wedge in plan)
    xa, xb = X_REAR - 1, X_FRONT + 1
    ya, yb = LAND_A + LAND_B * xa, LAND_A + LAND_B * xb
    plan = make_face(Polyline((xa, -ya), (xb, -yb), (xb, yb), (xa, ya), close=True))
    wedge = extrude(plan, amount=30, both=True)          # XY profile along Z
    plug = blk & wedge

    # through-bolt holes (axis Y)
    for (x, z) in (U18, U28):
        plug -= Pos(x, 0, z) * Rot(90, 0, 0) * Cylinder(BOLT_D / 2, 40)

    # jack bore along X at Z = JACK_Z
    ax = Rot(0, 90, 0)
    def xcyl(d, x0, x1):
        return Pos((x0 + x1) / 2, 0, JACK_Z) * ax * Cylinder(d / 2, x1 - x0)
    plug -= xcyl(CB_D, X_REAR - 1, CB_END)
    # teardrop crown toward +Y (the part prints lying on its -Y land face)
    r = CB_D / 2
    plug -= Pos((X_REAR - 1 + CB_END) / 2, 0, JACK_Z) * Rot(45, 0, 0) * Box(CB_END - X_REAR + 1, 2 * r, 2 * r) \
        & Pos((X_REAR - 1 + CB_END) / 2, 1.5 * r, JACK_Z) * Box(CB_END - X_REAR + 1, 3 * r, 3 * r)
    plug -= xcyl(TAP_D, CB_END - 0.5, NUT_X0 + 0.2)
    plug -= xcyl(CLR_D, NUT_X0 + NUT_T - 0.2, X_FRONT + 1)
    # nut pocket: flats facing +/-Y, drop-in from the top
    corner = NUT_AF / 0.866025 / 2          # circumradius
    pocket = Pos(NUT_X0 + NUT_T / 2, 0, JACK_Z - corner + (Z_TOP + 1 - (JACK_Z - corner)) / 2) * \
        Box(NUT_T, NUT_AF, Z_TOP + 1 - (JACK_Z - corner))
    plug -= pocket
    return plug

if __name__ == "__main__":
    p = build(sys.argv[1])
    bb = p.bounding_box()
    print("volume %.1f mm3, bbox" % p.volume, [round(v, 2) for v in (bb.min.X, bb.min.Y, bb.min.Z, bb.max.X, bb.max.Y, bb.max.Z)])
    export_step(p, sys.argv[2])
