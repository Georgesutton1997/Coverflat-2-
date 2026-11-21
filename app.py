"""CoverFlat backend: upload a STEP file, flatten each face into a 2D panel,
report stretch, and export a 1:1 DXF with seam allowance."""
import asyncio, base64, datetime, html, io, itertools, json, math, secrets, tempfile, time, uuid, os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.sparse.linalg import lsqr
from shapely.geometry import Polygon, Point, LineString, MultiLineString
from shapely.ops import unary_union, substring, linemerge
import cadquery as cq
import ezdxf
from contextlib import asynccontextmanager
from fastapi import Body, FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

KEEP = float(os.environ.get("COVERFLAT_KEEP_HOURS", "6")) * 3600  # uploads and jobs are deleted after this long unused
UP_DIR = os.path.join(tempfile.gettempdir(), "coverflat")
os.makedirs(UP_DIR, exist_ok=True)
TIMES: dict[str, float] = {}  # upload or job id -> last time used


def sweep():
    """Delete uploaded STEP files and stored jobs that have not been used for KEEP seconds."""
    cutoff = time.time() - KEEP
    for k, t in list(TIMES.items()):
        if t < cutoff:
            TIMES.pop(k, None)
            JOBS.pop(k, None)
            path = UPLOADS.pop(k, None)
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass


@asynccontextmanager
async def lifespan(_):
    for f in os.listdir(UP_DIR):  # files left behind by an earlier run
        try:
            os.unlink(os.path.join(UP_DIR, f))
        except OSError:
            pass

    async def loop():
        while True:
            await asyncio.sleep(600)
            sweep()
    task = asyncio.create_task(loop())
    yield
    task.cancel()


app = FastAPI(title="CoverFlat", lifespan=lifespan)
PASSWORD = os.environ.get("COVERFLAT_PASSWORD", "")  # set this when hosting online


@app.middleware("http")
async def basic_auth(request, call_next):
    """Ask for a password (any user name) when COVERFLAT_PASSWORD is set."""
    if PASSWORD:
        ok, auth = False, request.headers.get("authorization", "")
        if auth.startswith("Basic "):
            try:
                ok = secrets.compare_digest(base64.b64decode(auth[6:]).decode().partition(":")[2], PASSWORD)
            except Exception:
                pass
        if not ok:
            return Response("Password required", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="CoverFlat"'})
    return await call_next(request)
UPLOADS: dict[str, str] = {}  # upload id -> STEP file kept on disk
JOBS: dict[str, list[dict]] = {}  # job id -> panels (in-memory; use Redis/disk for production)
MAX_MB = 50


def lscm(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Least-squares conformal map of a 3D triangle mesh to 2D (units preserved)."""
    n = len(V)
    rows, cols, vals = [], [], []
    for t, (a, b, c) in enumerate(T):
        p0, p1, p2 = V[a], V[b], V[c]
        e1 = p1 - p0
        l1 = np.linalg.norm(e1)
        if l1 < 1e-12:
            continue
        x_ax = e1 / l1
        e2 = p2 - p0
        x2, y2 = e2 @ x_ax, np.linalg.norm(e2 - (e2 @ x_ax) * x_ax)
        pts = [(0.0, 0.0), (l1, 0.0), (x2, y2)]
        area2 = l1 * y2
        if area2 < 1e-12:
            continue
        s = 1.0 / np.sqrt(area2)
        for k, vid in zip(range(3), (a, b, c)):
            xn, yn = pts[(k + 1) % 3], pts[(k + 2) % 3]
            w = complex(xn[0] - yn[0], xn[1] - yn[1]) * s
            rows.append(t); cols.append(vid); vals.append(w)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(T), n), dtype=complex)

    # pin the two most distant vertices along the main axis
    c0 = V - V.mean(0)
    axis = np.linalg.svd(c0, full_matrices=False)[2][0]
    proj = c0 @ axis
    i0, i1 = int(proj.argmin()), int(proj.argmax())
    d = np.linalg.norm(V[i1] - V[i0])
    pinned = {i0: 0j, i1: complex(d, 0)}
    free = np.array([i for i in range(n) if i not in pinned])
    P = np.array(list(pinned)); up = np.array(list(pinned.values()))

    Af, Ap = A[:, free], A[:, P]
    rhs = -(Ap @ up)
    M = sp.bmat([[Af.real, -Af.imag], [Af.imag, Af.real]]).tocsr()
    b = np.concatenate([rhs.real, rhs.imag])
    sol = lsqr(M, b, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    k = len(free)
    U = np.zeros((n, 2))
    U[free, 0], U[free, 1] = sol[:k], sol[k:]
    U[P, 0], U[P, 1] = up.real, up.imag
    return U


def tri_areas(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    a, b, c = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
    cr = np.cross(b - a, c - a)
    return 0.5 * (np.abs(cr) if cr.ndim == 1 else np.linalg.norm(cr, axis=1))


def tri_areas_2d(U, T):
    a, b, c = U[T[:, 0]], U[T[:, 1]], U[T[:, 2]]
    return 0.5 * np.abs((b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1]) - (b[:, 1]-a[:, 1])*(c[:, 0]-a[:, 0]))


def flatten_piece(V, T):
    used = np.unique(T)
    Tl = np.searchsorted(used, T)
    Vl = V[used]
    U = lscm(Vl, Tl)
    a3 = tri_areas(Vl, Tl)
    U *= np.sqrt(a3.sum() / max(tri_areas_2d(U, Tl).sum(), 1e-12))
    st = tri_areas_2d(U, Tl) / np.maximum(a3, 1e-12)  # flat/3D area: 1.0 = no stretch
    return used, Tl, U, st, a3


def pick_notches(V, shared, spacing):
    """Choose seam vertices every `spacing` mm. Both panels sharing the seam get the same ones."""
    if len(shared) < 2 or spacing <= 0:
        return []
    P = V[shared]; c = P - P.mean(0)
    order = np.argsort(c @ np.linalg.svd(c, full_matrices=False)[2][0])
    P, ids = P[order], shared[order]
    out, run = [(int(ids[0]), int(ids[0]), 0.0)], 0.0
    for i in range(1, len(ids)):
        run += np.linalg.norm(P[i] - P[i - 1])
        if run >= spacing:
            out.append((int(ids[i]), int(ids[i]), 0.0)); run = 0.0
    return out


def split_pieces(V, T, max_dev, spacing, label, depth=0, notches=(), tags=(), ids=None):
    """Flatten; if too distorted, split the surface in two (a numbered seam) and retry."""
    used, Tl, U, st, a3 = flatten_piece(V, T)
    dev = np.abs(st - 1)
    closed = is_closed(T)  # a closed shell has no edge to unroll from, so it must be cut open
    cen = V[T].mean(1); c = cen - cen.mean(0)
    s = c @ np.linalg.svd(c, full_matrices=False)[2][0]
    m = s <= np.median(s)
    stop = depth >= 6 or len(T) < 40 or m.all() or not m.any()
    if stop or (not closed and (max_dev <= 0 or np.percentile(dev, 95) <= max_dev)):
        have = set(used.tolist())
        return [dict(label=label, used=used, T=Tl, U=U, stretch=st, a3=a3, dev=float(dev.max()),
                     notches=[r for r in notches if r[0] in have and r[1] in have], tags=[t for t in tags if t[0] in have])]
    shared = np.intersect1d(T[m].ravel(), T[~m].ravel())
    new = pick_notches(V, shared, spacing)
    ntag = []
    if ids is not None and len(shared) >= 2:
        pr = V[shared] - V[shared].mean(0)
        o = np.argsort(pr @ np.linalg.svd(pr, full_matrices=False)[2][0])
        n = next(ids)
        ntag = [(int(shared[o[0]]), f"{n}a"), (int(shared[o[-1]]), f"{n}b")]
    out = []
    for part, suffix in ((T[m], "A"), (T[~m], "B")):
        out += split_pieces(V, part, max_dev, spacing, label + suffix, depth + 1,
                            list(notches) + new, list(tags) + ntag, ids)
    return out


def is_closed(T):
    e = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return bool((cnt == 2).all())


def mesh_body(body, tol):
    """Mesh a whole body at once so faces share edge points, then weld duplicate vertices."""
    verts, tris = body.tessellate(tol, 0.2)
    if len(tris) < 2:
        return None
    V = np.array([[v.x, v.y, v.z] for v in verts]); T = np.array(tris)
    keys = np.round(V / 1e-3).astype(np.int64)
    _, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    V, T = V[first], inv.ravel()[T]
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
    return V, T


def components(T, n):
    """Group triangles into connected pieces."""
    g = sp.coo_matrix((np.ones(2 * len(T)), (np.r_[T[:, 0], T[:, 1]], np.r_[T[:, 1], T[:, 2]])), shape=(n, n))
    k, lab = connected_components(g, directed=False)
    return k, lab[T[:, 0]]


def boundary_samples(V, T, step):
    """Points every `step` mm along the free edges of a mesh. Straight edges have no interior
    mesh vertices, so vertices alone are too sparse to match. Returns (points, refs) where each
    ref is (vertex a, vertex b, t): the point lies at fraction t along edge a-b."""
    e = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    pts, refs = [], []
    for va, vb in u[cnt == 1]:
        n = max(int(np.linalg.norm(V[vb] - V[va]) / step), 1)
        for k in range(n + 1):
            t = k / n
            pts.append(V[va] + t * (V[vb] - V[va])); refs.append((int(va), int(vb), t))
    return np.array(pts).reshape(-1, 3), refs


def seam_notches(bodies, dist, spacing, ids):
    """Find free edges of different bodies within `dist` mm of each other (edges that will be
    sewn). Each such seam gets a number, matching notches on both sides and end labels."""
    extra = [[] for _ in bodies]; tags = [[] for _ in bodies]
    if dist <= 0:
        return extra, tags
    trees = [cKDTree(b["bp"]) if len(b["bp"]) else None for b in bodies]
    end_vertex = lambda r: r[0] if r[2] < 0.5 else r[1]
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            if trees[j] is None or len(bodies[i]["bp"]) == 0:
                continue
            d, k = trees[j].query(bodies[i]["bp"], distance_upper_bound=dist)
            ok = np.isfinite(d)
            if ok.sum() < 2:
                continue
            idx, kj = np.where(ok)[0], k[ok]
            P = bodies[i]["bp"][idx]
            c = P - P.mean(0)
            order = np.argsort(c @ np.linalg.svd(c, full_matrices=False)[2][0])
            gaps = np.linalg.norm(np.diff(P[order], axis=0), axis=1)
            for run in np.split(order, np.where(gaps > 5 * dist)[0] + 1):
                if len(run) < 2:
                    continue
                n = next(ids)
                for end, name in ((run[0], f"{n}a"), (run[-1], f"{n}b")):
                    tags[i].append((end_vertex(bodies[i]["bref"][idx[end]]), name))
                    tags[j].append((end_vertex(bodies[j]["bref"][kj[end]]), name))
                acc, last = spacing, None
                for o in run:
                    if last is not None:
                        acc += np.linalg.norm(P[o] - last)
                    if spacing > 0 and acc >= spacing:
                        extra[i].append(bodies[i]["bref"][idx[o]]); extra[j].append(bodies[j]["bref"][kj[o]])
                        acc = 0.0
                    last = P[o]
    return extra, tags


def step_bodies(path, mode):
    """Read a STEP file with its colours. Returns [(cq shape, ['#rrggbb', ...]), ...] (all colours found)."""
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorSurf, XCAFDoc_ColorGen
    from OCP.TDF import TDF_LabelSequence
    from OCP.Quantity import Quantity_Color
    from OCP.IFSelect import IFSelect_RetDone
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    XCAFApp_Application.GetApplication_s().NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)
    rd = STEPCAFControl_Reader(); rd.SetColorMode(True)
    if rd.ReadFile(path) != IFSelect_RetDone:
        raise ValueError("unreadable STEP file")
    rd.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main()); ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    labels = TDF_LabelSequence(); st.GetFreeShapes(labels)

    def colours(sh):
        """Every distinct colour stored on this shape (surface colour, then generic colour)."""
        found = []
        for kind in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
            col = Quantity_Color()
            if ct.GetColor_s(sh, kind, col):
                hx = "#%02x%02x%02x" % tuple(round(max(0.0, min(1.0, v)) * 255)
                                              for v in (col.Red(), col.Green(), col.Blue()))
                if hx not in found:
                    found.append(hx)
        return found

    out = []
    for i in range(1, labels.Length() + 1):
        top = cq.Shape.cast(st.GetShape_s(labels.Value(i)))
        parts = top.Faces() if mode == "face" else (top.Solids() or top.Shells() or top.Faces())
        for b in parts:
            cs = colours(b.wrapped)
            for f in sorted(b.Faces(), key=lambda f: -f.Area()):  # largest faces first
                cs += [c for c in colours(f.wrapped) if c not in cs]
            out.append((b, cs))
    return out


def fold_edges(V, T, min_angle):
    """Mesh edges where the surface bends sharply (a fold). Returns [(va, vb, angle_deg)].
    Uses |dot| of the face normals so triangle winding does not matter."""
    if min_angle <= 0 or len(T) < 2:
        return []
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    E = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    tid = np.tile(np.arange(len(T)), 3)
    o = np.lexsort((E[:, 1], E[:, 0]))
    E, tid = E[o], tid[o]
    same = (E[1:] == E[:-1]).all(axis=1)
    i = np.where(same & ~np.r_[False, same[:-1]] & ~np.r_[same[1:], False])[0]  # edges shared by exactly 2 triangles
    ang = np.degrees(np.arccos(np.abs((n[tid[i]] * n[tid[i + 1]]).sum(1)).clip(0, 1)))
    keep = ang >= min_angle
    return [(int(E[k, 0]), int(E[k, 1]), float(a)) for k, a in zip(i[keep], ang[keep])]


def read_parts(path, mode):
    try:
        return step_bodies(path, mode)
    except Exception:  # colours unavailable: read geometry only
        shape = cq.importers.importStep(path)
        if mode == "face":
            vals = shape.faces().vals()
        else:  # solids first, then surface bodies (shells), then loose faces
            vals = shape.solids().vals() or shape.shells().vals() or shape.faces().vals()
        return [(v, []) for v in vals]


def build_panels(step_path, tol, min_area, max_dev, spacing, mode="body", match=10.0, fold=30.0, overrides=None):
    parts = read_parts(step_path, mode)
    overrides = overrides or {}
    bodies = []
    for idx, (body, colour) in enumerate(parts):
        if overrides.get(idx):  # colour picked by the user
            colour = [overrides[idx]] + [c for c in colour if c != overrides[idx]]
        meshed = mesh_body(body, tol)
        if meshed is None:
            continue
        V, T = meshed
        k, lab = components(T, len(V))
        for ci in range(k):
            Tc = T[lab == ci]
            if len(Tc) < 2 or tri_areas(V, Tc).sum() < min_area:
                continue
            bp, bref = boundary_samples(V, Tc, max(match / 2, 1.0)) if match > 0 else (np.zeros((0, 3)), [])
            bodies.append(dict(name=str(idx) if k == 1 else f"{idx}.{ci}", V=V, T=Tc, bp=bp, bref=bref, folds=fold_edges(V, Tc, fold),
                               color=colour[0] if colour else None,
                               color2=colour[1] if len(colour) > 1 else None))
    ids = itertools.count(1)
    extra, tags = seam_notches(bodies, match, spacing, ids)
    panels = []
    for bd, ex, tg in zip(bodies, extra, tags):
        for pc in split_pieces(bd["V"], bd["T"], max_dev, spacing, bd["name"], 0, ex, tg, ids):
            U, Tl = pc["U"], pc["T"]
            outline = unary_union([Polygon(U[t]) for t in Tl if Polygon(U[t]).area > 0])
            if outline.geom_type == "MultiPolygon":
                outline = max(outline.geoms, key=lambda g: g.area)
            pc.update(outline=outline, area3d=float(pc["a3"].sum()), max_dev=pc["dev"], nbodies=len(bodies), color=bd["color"], color2=bd["color2"])
            have = set(pc["used"].tolist())
            pc["folds"] = [f for f in bd["folds"] if f[0] in have and f[1] in have]
            panels.append(pc)
    return panels


def layout(panels, gap=40.0, row_width=3000.0):
    x = y = row_h = 0.0
    for p in panels:
        minx, miny, maxx, maxy = p["outline"].bounds
        w, h = maxx - minx, maxy - miny
        if x + w > row_width and x > 0:
            x, y, row_h = 0.0, y + row_h + gap, 0.0
        off = np.array([x - minx, y - miny])
        p["U"] = p["U"] + off
        p["outline"] = Polygon(np.array(p["outline"].exterior.coords) + off)
        x += w + gap
        row_h = max(row_h, h)


def job_panels(job):
    p = JOBS.get(job)
    if p:
        TIMES[job] = time.time()  # using a job keeps it alive
    return p


async def stash(file, upload):
    """Keep the uploaded STEP file on disk so it can be re-read without uploading again."""
    if upload:
        if upload not in UPLOADS:
            raise HTTPException(404, "Upload expired. Choose the file again.")
        TIMES[upload] = time.time()
        return upload, UPLOADS[upload]
    if file is None or not file.filename.lower().endswith((".stp", ".step")):
        raise HTTPException(400, "Upload a .stp or .step file.")
    data = await file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_MB} MB.")
    with tempfile.NamedTemporaryFile(suffix=".stp", delete=False, dir=UP_DIR) as f:
        f.write(data)
    uid = uuid.uuid4().hex
    UPLOADS[uid] = f.name
    TIMES[uid] = time.time()
    return uid, f.name


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(None), upload: str = Form(""), mode: str = Form("body"),
                  tolerance: float = Form(2.0)):
    """Coarse meshes and colours of every body, for the 3D viewer."""
    uid, path = await stash(file, upload)
    try:
        parts = read_parts(path, mode)
    except Exception as e:
        raise HTTPException(422, f"Could not read the STEP file: {e}")
    out = []
    for idx, (body, cs) in enumerate(parts):
        m = mesh_body(body, max(tolerance, 1.0))
        if m is None:
            continue
        V, T = m
        out.append(dict(idx=idx, colors=cs[:2], pos=np.round(V, 1).ravel().tolist(), tris=T.ravel().tolist()))
    return {"upload": uid, "bodies": out}


@app.post("/api/arrange/{job}")
def arrange(job: str, body: dict = Body(...)):
    """Move, rotate or flip one panel. State is absolute: flip, then rotate about the centroid, then shift."""
    panels = job_panels(job)
    if not panels:
        raise HTTPException(404, "Job expired. Upload the file again.")
    for p in panels:
        if p["label"] == body.get("label"):
            c, r = p["c0"], math.radians(body.get("a", 0))
            R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])

            def tf(X):
                Y = X - c
                if body.get("f"):
                    Y = Y * np.array([-1.0, 1.0])
                return Y @ R.T + c + np.array([body.get("x", 0.0), body.get("y", 0.0)])
            p["U"] = tf(p["U0"])
            p["outline"] = Polygon(tf(np.array(p["outline0"].exterior.coords)))
            return {"ok": True}
    raise HTTPException(404, "Unknown panel.")


@app.post("/api/flatten")
async def flatten(file: UploadFile = File(None), upload: str = Form(""), colors: str = Form(""),
                  tolerance: float = Form(2.0), seam_allowance: float = Form(15.0),
                  max_stretch: float = Form(8.0), notch_spacing: float = Form(200.0),
                  grain_angle: float = Form(0.0), mode: str = Form("body"),
                  match_dist: float = Form(10.0), fold_angle: float = Form(30.0)):
    uid, path = await stash(file, upload)
    try:
        ov = {int(k): v for k, v in json.loads(colors or "{}").items()}
        panels = build_panels(path, tolerance, 100.0, max_stretch / 100.0, notch_spacing, mode,
                              match_dist, fold_angle, ov)
    except Exception as e:
        raise HTTPException(422, f"Could not read the STEP file: {e}")
    if not panels:
        raise HTTPException(422, "No surfaces found in the file.")
    layout(panels)
    for p in panels:
        p["seam"], p["grain"] = seam_allowance, grain_angle
        p["U0"], p["outline0"] = p["U"].copy(), p["outline"]
        p["c0"] = np.array(p["outline"].centroid.coords[0])
    job = uuid.uuid4().hex
    JOBS[job] = panels
    TIMES[job] = time.time()
    return {"job": job, "seams": len(seam_summary(panels)), "bodies": panels[0]["nbodies"], "folds": sum(len(fold_lines(p)) for p in panels), "coloured": sum(p["color"] is not None for p in panels), "panels": [dict(
        id=p["label"], cx=float(p["c0"][0]), cy=float(p["c0"][1]), color=p["color"], color2=p["color2"], area_mm2=round(p["area3d"], 1), max_stretch_pct=round(p["max_dev"] * 100, 1),
        verts=np.round(p["U"], 2).tolist(), tris=p["T"].tolist(),
        stretch=np.round(p["stretch"], 3).tolist(),
        outline=np.round(np.array(p["outline"].exterior.coords), 2).tolist(),
        folds=[[np.round(c, 1).tolist() for c in ln] for ln in fold_lines(p)],
        fold_max=round(max([f[2] for f in p["folds"]], default=0)),
        notches=[[np.round(a, 1).tolist(), np.round(b, 1).tolist()] for a, b in notch_lines(p)],
        tags=[[*np.round(q, 1).tolist(), t, seam_color(int(t[:-1]))] for _, q, t in tag_points(p)],
        edges=[[n, np.round(pts, 1).tolist(), seam_color(n)] for n, pts in seam_edges(p)],
    ) for p in panels]}


def notch_lines(p):
    o = p["outline"]; ring = o.exterior
    length = p["seam"] if p["seam"] > 0 else 5.0
    for va, vb, t in p["notches"]:
        ia, ib = int(np.searchsorted(p["used"], va)), int(np.searchsorted(p["used"], vb))
        d = ring.project(Point(p["U"][ia] + t * (p["U"][ib] - p["U"][ia])))
        pt, a, b = ring.interpolate(d), ring.interpolate(max(d - 0.5, 0)), ring.interpolate(min(d + 0.5, ring.length))
        t = np.array([b.x - a.x, b.y - a.y])
        if np.linalg.norm(t) < 1e-9:
            continue
        n = np.array([-t[1], t[0]]) / np.linalg.norm(t)
        P = np.array([pt.x, pt.y])
        if o.contains(Point(*(P + n * 0.5))):
            n = -n
        yield P, P + n * length


def pts_str(pts):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)


def fold_lines(p):
    """Fold edges of a panel as merged polylines in flat (2D) coordinates."""
    segs = []
    for va, vb, _ in p["folds"]:
        ia, ib = int(np.searchsorted(p["used"], va)), int(np.searchsorted(p["used"], vb))
        segs.append(LineString([p["U"][ia], p["U"][ib]]))
    if not segs:
        return []
    m = linemerge(MultiLineString(segs))
    return [list(g.coords) for g in (m.geoms if hasattr(m, "geoms") else [m])]


def fold_text(p):
    n = len(fold_lines(p))
    return f"{n} ({max(f[2] for f in p['folds']):.0f}&deg;)" if n else "-"


def tag_points(p):
    """Seam-end labels: (point on edge, label position just inside the panel, text)."""
    c = p["outline"].centroid; C = np.array([c.x, c.y])
    for g, text in p["tags"]:
        i = int(np.searchsorted(p["used"], g))
        P = p["U"][i]; v = C - P; n = np.linalg.norm(v)
        yield P, (P + v / n * 30 if n > 1e-9 else P), text


def grain_line(p):
    o = p["outline"]; c = o.centroid
    x0, y0, x1, y1 = o.bounds
    half = 0.2 * math.hypot(x1 - x0, y1 - y0)
    ang = math.radians(p["grain"])
    d = np.array([math.cos(ang), math.sin(ang)])
    C = np.array([c.x, c.y])
    return C - d * half, C + d * half, d


@app.get("/api/dxf/{job}")
def dxf(job: str, labels: int = 0):
    panels = job_panels(job)
    if not panels:
        raise HTTPException(404, "Job expired. Upload the file again.")
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    for name, col in (("SEWN", 3), ("CUT", 1), ("NOTCH", 5), ("GRAIN", 6), ("LABEL", 7), ("SEAMID", 2)):
        doc.layers.add(name, color=col)
    doc.layers.add("FOLD", color=5, linetype="DASHED")
    msp = doc.modelspace()
    for p in panels:
        o = p["outline"]
        msp.add_lwpolyline(list(o.exterior.coords), close=True, dxfattribs={"layer": "SEWN"})
        if p["seam"] > 0:
            cut = o.buffer(p["seam"], join_style=2)
            msp.add_lwpolyline(list(cut.exterior.coords), close=True, dxfattribs={"layer": "CUT"})
        for pts in fold_lines(p):
            msp.add_lwpolyline(pts, dxfattribs={"layer": "FOLD"})
        for a, b in notch_lines(p):
            msp.add_line(tuple(a), tuple(b), dxfattribs={"layer": "NOTCH"})
        for P, Q, text in (tag_points(p) if labels else []):
            rgb = seam_rgb(int(text[:-1]))
            msp.add_circle(tuple(P), 4, dxfattribs={"layer": "SEAMID", "true_color": rgb})
            msp.add_text(text, height=18, dxfattribs={"layer": "SEAMID", "insert": tuple(Q), "true_color": rgb})
        for n, pts in seam_edges(p):
            msp.add_lwpolyline(pts, dxfattribs={"layer": "SEAMID", "true_color": seam_rgb(n), "lineweight": 50})
        a, b, d = grain_line(p)
        msp.add_line(tuple(a), tuple(b), dxfattribs={"layer": "GRAIN"})
        for tip, sgn in ((a, -1), (b, 1)):
            for w in (math.radians(25), -math.radians(25)):
                r = np.array([[math.cos(w), -math.sin(w)], [math.sin(w), math.cos(w)]]) @ (d * -sgn)
                msp.add_line(tuple(tip), tuple(tip + r * 25), dxfattribs={"layer": "GRAIN"})
        cc = o.centroid
        msp.add_text(f"P{p['label']} CUT 1", height=20,
                     dxfattribs={"layer": "LABEL", "insert": (cc.x, cc.y + 30)})
    buf = io.StringIO(); doc.write(buf)
    return Response(buf.getvalue(), media_type="application/dxf",
                    headers={"Content-Disposition": "attachment; filename=cover_panels.dxf"})


SEAM_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008b8b",
               "#9a6324", "#d2008a", "#6b8e00", "#000075", "#808080", "#c9a000"]


def seam_color(n):
    return SEAM_COLORS[(n - 1) % len(SEAM_COLORS)]


def seam_rgb(n):
    c = seam_color(n)
    return ezdxf.colors.rgb2int((int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)))


def seam_edges(p):
    """The stretch of outline between a seam's two end labels, as (seam number, points)."""
    ring = p["outline"].exterior; L = ring.length
    ends = {}
    for g, text in p["tags"]:
        i = int(np.searchsorted(p["used"], g))
        ends.setdefault(int(text[:-1]), {})[text[-1]] = ring.project(Point(p["U"][i]))
    for n, e in ends.items():
        if "a" not in e or "b" not in e:
            continue
        lo, hi = sorted((e["a"], e["b"]))
        if hi - lo <= L / 2:
            pts = list(substring(ring, lo, hi).coords)
        else:
            pts = list(substring(ring, hi, L).coords) + list(substring(ring, 0, lo).coords)
        if len(pts) >= 2:
            yield n, pts


def seam_summary(panels):
    """{seam number: [(panel label, edge length mm or None), ...]} from the a/b end labels."""
    seams = {}
    for p in panels:
        ring = p["outline"].exterior
        ends = {}
        for g, text in p["tags"]:
            i = int(np.searchsorted(p["used"], g))
            ends.setdefault(text[:-1], {})[text[-1]] = ring.project(Point(p["U"][i]))
        for n, e in ends.items():
            ln = None
            if "a" in e and "b" in e:
                dd = abs(e["a"] - e["b"]); ln = min(dd, ring.length - dd)
            seams.setdefault(int(n), []).append((p["label"], ln))
    return dict(sorted(seams.items()))


WI_CSS = """body{font:14px/1.5 'Segoe UI',Arial,sans-serif;color:#111;max-width:900px;margin:24px auto;padding:0 16px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:26px 0 8px;border-bottom:2px solid #111;padding-bottom:3px}
table{border-collapse:collapse;width:100%;margin:6px 0}th,td{border:1px solid #888;padding:5px 8px;text-align:left}
th{background:#eee}.meta td{border:0;padding:2px 12px 2px 0}svg{width:100%;height:auto;border:1px solid #bbb}
li{margin:6px 0}.warn{color:#a33}@media print{*{-webkit-print-color-adjust:exact;print-color-adjust:exact}h2{break-after:avoid}tr,li{break-inside:avoid}}"""


def pick(p, swap):
    return p["color2"] if swap and p["color2"] else p["color"]


@app.get("/api/instructions/{job}", response_class=HTMLResponse)
def instructions(job: str, swap: int = 0, job_name: str = "", fabric: str = "", thread: str = "", stitch: str = "", labels: int = 0):
    panels = job_panels(job)
    if not panels:
        raise HTTPException(404, "Job expired. Upload the file again.")
    e = lambda v, d="To be specified": html.escape(v) if v.strip() else d
    sa = panels[0]["seam"]
    seams = seam_summary(panels)

    # layout drawing
    x0 = min(p["outline"].bounds[0] for p in panels); y0 = min(p["outline"].bounds[1] for p in panels)
    x1 = max(p["outline"].bounds[2] for p in panels); y1 = max(p["outline"].bounds[3] for p in panels)
    pad = sa + 40; fs = (x1 - x0) / 60; svg = ""
    for p in panels:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in p["outline"].exterior.coords)
        c = p["outline"].centroid
        svg += f'<polygon points="{pts}" fill="{pick(p, swap) or "#f4f4f4"}" stroke="#111" stroke-width="1.5" vector-effect="non-scaling-stroke"/>'
        for pts in fold_lines(p):
            svg += f'<polyline points="{pts_str(pts)}" fill="none" stroke="#111" stroke-width="3" stroke-dasharray="10 6" vector-effect="non-scaling-stroke"/>'
        for n, pts in seam_edges(p):
            svg += f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{seam_color(n)}" stroke-width="5" stroke-linecap="round" vector-effect="non-scaling-stroke"/>'
        for a, b in notch_lines(p):
            svg += f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="#e67e22" stroke-width="2" vector-effect="non-scaling-stroke"/>'
        svg += f'<text x="{c.x:.1f}" y="{c.y:.1f}" font-size="{fs*1.6:.1f}" text-anchor="middle" fill="#777">P{html.escape(p["label"])}</text>'
        for _, q, t in (tag_points(p) if labels else []):
            svg += f'<text x="{q[0]:.1f}" y="{q[1]:.1f}" font-size="{fs:.1f}" font-weight="700" fill="{seam_color(int(t[:-1]))}" stroke="#fff" stroke-width="{fs/8:.1f}" paint-order="stroke" text-anchor="middle" dominant-baseline="middle">{t}</text>'
    svg = f'<svg viewBox="{x0-pad:.0f} {y0-pad:.0f} {x1-x0+2*pad:.0f} {y1-y0+2*pad:.0f}">{svg}</svg>'

    cut = "".join(
        f'<tr><td>P{html.escape(p["label"])}</td><td><span style="display:inline-block;width:16px;height:16px;background:{pick(p, swap) or "#fff"};border:1px solid #888;vertical-align:middle"></span> {pick(p, swap) or "-"}</td><td>1</td>'
        f'<td>{p["outline"].bounds[2]-p["outline"].bounds[0]:.0f} x {p["outline"].bounds[3]-p["outline"].bounds[1]:.0f}</td>'
        f'<td>{p["area3d"]/1e6:.3f}</td><td>{p["grain"]:.0f}&deg;</td><td>{len(p["notches"])}</td><td>{fold_text(p)}</td>'
        f'<td>{p["dev"]*100:.1f}%</td></tr>' for p in panels)

    rows, steps = "", ""
    foldstep = ("<li>Mark each dashed fold line on the wrong side and form the fold (crease or press) before joining seams.</li>"
                if any(p["folds"] for p in panels) else "")
    for n, sides in seams.items():
        names = " + ".join("P" + html.escape(l) for l, _ in sides)
        lens = [ln for _, ln in sides if ln]
        lt = " / ".join(f"{ln:.0f}" if ln else "see drawing" for _, ln in sides)
        diff = (max(lens) - min(lens)) / min(lens) * 100 if len(lens) >= 2 else None
        dt = f"{diff:.1f}%" if diff is not None else "-"
        note = ""
        if diff is not None and diff > 3:
            note = f' <span class="warn">Edges differ by {diff:.1f}%. Ease the longer edge evenly between notches.</span>'
        sw = f'<span style="display:inline-block;width:16px;height:16px;background:{seam_color(n)};margin-right:6px;vertical-align:middle"></span>'
        rows += f"<tr><td>{sw}{n}</td><td>{names}</td><td>{lt}</td><td>{dt}</td></tr>"
        ends = f" Match <b>{n}a</b> to <b>{n}a</b> and <b>{n}b</b> to <b>{n}b</b>." if labels else ""
        steps += (f"<li>{sw}<b>Seam {n}:</b> join {names}.{ends} Match the edges of the same colour and all notches, "
                  f"pin or clip, then sew at {sa:.0f} mm.{note}</li>")
    if not seams:
        rows = '<tr><td colspan="4">No seams were detected in this job.</td></tr>'

    body = f"""<h1>Sewing work instruction</h1>
<table class="meta"><tr><td><b>Job</b></td><td>{e(job_name, 'Untitled')}</td><td><b>Date</b></td><td>{datetime.date.today():%d %b %Y}</td></tr>
<tr><td><b>Fabric</b></td><td>{e(fabric)}</td><td><b>Thread</b></td><td>{e(thread)}</td></tr>
<tr><td><b>Stitch</b></td><td>{e(stitch)}</td><td><b>Seam allowance</b></td><td>{sa:.0f} mm</td></tr></table>
<h2>1. Cutting list</h2>
<p>Cut each panel along the outer cut line (seam allowance included). Mark the grain line, notches and seam labels on every panel.</p>
<table><tr><th>Panel</th><th>Body colour (STEP)</th><th>Cut qty</th><th>Size W x H (mm)</th><th>Area (m&sup2;)</th><th>Grain</th><th>Notches</th><th>Folds</th><th>Max stretch</th></tr>{cut}</table>
<h2>2. Panel layout</h2>{svg}
<p>Edges drawn in the same colour are sewn together. {'Labels such as 3a and 3b are the ends of seam 3. ' if labels else ''}Orange ticks are notches. Dashed black lines are folds.</p>
<h2>3. Seam list</h2>
<table><tr><th>Seam</th><th>Joins</th><th>Edge length (mm)</th><th>Difference</th></tr>{rows}</table>
<p>Lengths are measured on the flat pattern between the seam end points, so treat them as approximate on curved edges.</p>
<h2>4. Sewing steps</h2>
<ol><li>Check every panel against the cutting list. Transfer notches and seam end labels to the wrong side.</li>
{foldstep}{steps}
<li>Press seams as specified for the fabric. Trim loose threads.</li></ol>
<p>The seams are listed in number order. Change the order to suit your process.</p>
<h2>5. Final check</h2>
<p>&#9744; All seams sewn and all notches matched &nbsp; &#9744; Cover dry-fitted on the boat &nbsp; &#9744; Dimensions checked against the drawing</p>"""
    return HTMLResponse(f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>Sewing work instruction</title><style>{WI_CSS}</style></head><body>{body}</body></html>")


@app.get("/")
def index():
    base = os.path.dirname(__file__)
    for d in ("static", "Static"):  # Windows may have saved the folder as "Static"
        p = os.path.join(base, d, "index.html")
        if os.path.exists(p):
            return FileResponse(p)
    raise HTTPException(404, "index.html was not found in the static folder.")
