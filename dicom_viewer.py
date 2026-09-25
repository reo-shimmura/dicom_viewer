#!/usr/bin/env python3
"""DICOM CT ビューワ (PySide6 + pydicom + numpy)

処理の流れ (README の最低4処理)
  1. フォルダ内のファイルを読み込み、シリーズ (SeriesInstanceUID) ごとに分類する
  2. スライスを Image Position (Patient) を法線ベクトルへ射影した位置順に並べる
  3. 画素値を CT 値 (HU = 画素値 x RescaleSlope + RescaleIntercept) に換算する
  4. CT 値をウィンドウ幅/レベルで 0-255 の明るさに変換して表示する
追加機能: MPR (冠状断/矢状断)、距離・ROI 計測、メタデータ重畳表示、タグ一覧 など

使い方: python dicom_viewer.py [DICOMフォルダ]
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pydicom
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF, Qt, Signal

# 名前, WC, WW (None は DICOM ファイル記載の既定値)
PRESETS = [
    ("DICOM既定", None, None),
    ("脳", 40, 80),
    ("硬膜下", 75, 215),
    ("軟部組織", 40, 400),
    ("骨", 600, 2800),
    ("肺", -600, 1500),
]


# ======================================================================
# 1-3. 読み込み・並べ替え・CT値換算
# ======================================================================
@dataclass
class Series:
    uid: str
    number: str
    description: str
    datasets: list                # 位置順に並んだ pydicom Dataset (PixelData は破棄済み)
    volume: np.ndarray            # CT値 (nz, ny, nx)
    ipps: np.ndarray              # 各スライスの Image Position (nz, 3)
    positions: np.ndarray         # 法線方向の位置 [mm] (nz,)
    row_dir: np.ndarray           # 列が増える方向の単位ベクトル
    col_dir: np.ndarray           # 行が増える方向の単位ベクトル
    normal: np.ndarray
    sx: float                     # 列方向の画素間隔 [mm]
    sy: float                     # 行方向の画素間隔 [mm]
    dz: float                     # スライス間隔 [mm]
    default_window: tuple
    warnings: list = field(default_factory=list)

    @property
    def shape(self):
        return self.volume.shape

    def patient_point(self, z, y, x):
        """ボクセル (z,y,x) の患者座標 (LPS, mm)"""
        return self.ipps[z] + x * self.sx * self.row_dir + y * self.sy * self.col_dir


def _first(v, default=None):
    if v is None or v == "":
        return default
    try:
        if isinstance(v, (list, tuple, pydicom.multival.MultiValue)):
            v = v[0]
        return float(v)
    except (TypeError, ValueError):
        return default


def build_series(datasets: list) -> Series:
    """2. 位置順に並べ、3. CT値のボリュームを作る"""
    ds0 = datasets[0]
    warnings = []
    shape = (int(ds0.Rows), int(ds0.Columns))
    kept = [d for d in datasets if (int(d.Rows), int(d.Columns)) == shape]
    if len(kept) != len(datasets):
        warnings.append(f"画像サイズが異なる {len(datasets) - len(kept)} 枚を除外しました")
    datasets = kept

    iop = np.array(ds0.ImageOrientationPatient, dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    pos = np.array([float(np.dot(np.array(d.ImagePositionPatient, dtype=float), normal)) for d in datasets])
    order = np.argsort(pos, kind="stable")          # InstanceNumber ではなく空間位置で並べる
    datasets = [datasets[i] for i in order]
    pos = pos[order]
    ipps = np.array([d.ImagePositionPatient for d in datasets], dtype=float)

    diffs = np.diff(pos)
    diffs = diffs[diffs > 1e-3]
    if len(diffs):
        dz = float(np.median(diffs))
        if np.any(np.abs(diffs - dz) > 0.1 * dz):
            warnings.append("スライス間隔が不均一です (MPR は等間隔と仮定して表示)")
    else:
        dz = _first(getattr(ds0, "SliceThickness", None), 1.0)
    if abs(normal[2]) < 0.99:
        warnings.append("斜位断面のデータです (向き表示・MPR は近似)")

    ps = [float(v) for v in getattr(ds0, "PixelSpacing", [1.0, 1.0])]
    sy, sx = ps[0], ps[1]                            # PixelSpacing = [行間隔, 列間隔]

    slopes = [float(getattr(d, "RescaleSlope", 1)) for d in datasets]
    inters = [float(getattr(d, "RescaleIntercept", 0)) for d in datasets]
    as_int = all(s == 1 for s in slopes) and all(float(b).is_integer() for b in inters)
    vol = np.empty((len(datasets),) + shape, dtype=np.int16 if as_int else np.float32)
    for k, d in enumerate(datasets):
        hu = d.pixel_array.astype(np.float32) * slopes[k] + inters[k]   # HU = 画素値 x slope + intercept
        vol[k] = np.clip(hu, -32768, 32767) if as_int else hu
        try:
            del d.PixelData                          # メモリ節約 (画素はボリュームに保持済み)
        except AttributeError:
            pass

    wc, ww = _first(getattr(ds0, "WindowCenter", None)), _first(getattr(ds0, "WindowWidth", None))
    if wc is None or ww is None:
        lo, hi = np.percentile(vol[len(vol) // 2], [1, 99])
        wc, ww = (lo + hi) / 2, max(hi - lo, 1)
    return Series(
        uid=str(ds0.SeriesInstanceUID), number=str(getattr(ds0, "SeriesNumber", "")),
        description=str(getattr(ds0, "SeriesDescription", "")), datasets=datasets, volume=vol,
        ipps=ipps, positions=pos, row_dir=row_dir, col_dir=col_dir, normal=normal,
        sx=sx, sy=sy, dz=dz, default_window=(wc, ww), warnings=warnings)


def read_folder(folder: str, progress=None) -> list[Series]:
    """1. フォルダ内の DICOM を読み、シリーズごとに分類する。progress(i, n) が False を返すと中断"""
    paths = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(folder) for f in fs)
    groups: dict[str, list] = {}
    for i, p in enumerate(paths):
        if progress and not progress(i, len(paths)):
            return []
        try:
            ds = pydicom.dcmread(p)
        except Exception:                            # .DS_Store など DICOM 以外は無視
            continue
        if "PixelData" not in ds or "ImagePositionPatient" not in ds:
            continue
        groups.setdefault(str(ds.SeriesInstanceUID), []).append(ds)
    series = [build_series(g) for g in groups.values()]
    series.sort(key=lambda s: (int(s.number) if s.number.isdigit() else 0, s.uid))
    return series


# ======================================================================
# 共有状態
# ======================================================================
class State(QtCore.QObject):
    series_changed = Signal()
    cross_changed = Signal()
    window_changed = Signal()
    style_changed = Signal()
    proj_changed = Signal()

    def __init__(self):
        super().__init__()
        self.series: Series | None = None
        self.wc, self.ww = 40.0, 250.0
        self.cross = (0, 0, 0)          # (z, y, x)
        self.tool = "cross"
        self.invert = False
        self.show_cross = True
        self.show_patient = False       # 患者名・ID・生年月日は既定で非表示
        self.proj_mode = "none"         # none / mip / minip / mean (スラブ投影)
        self.slab_mm = 10.0

    def set_projection(self, mode=None, slab_mm=None):
        self.proj_mode = mode or self.proj_mode
        self.slab_mm = float(slab_mm) if slab_mm else self.slab_mm
        self.proj_changed.emit()

    def set_series(self, s: Series):
        self.series = s
        nz, ny, nx = s.shape
        self.cross = (nz // 2, ny // 2, nx // 2)
        self.wc, self.ww = s.default_window
        self.series_changed.emit()
        self.window_changed.emit()

    def set_cross(self, z, y, x):
        nz, ny, nx = self.series.shape
        c = (int(min(max(z, 0), nz - 1)), int(min(max(y, 0), ny - 1)), int(min(max(x, 0), nx - 1)))
        if c != self.cross:
            self.cross = c
            self.cross_changed.emit()

    def set_window(self, wc, ww):
        self.wc, self.ww = float(wc), max(float(ww), 1.0)
        self.window_changed.emit()

    def to_display(self, a: np.ndarray) -> np.ndarray:
        """4. CT値 -> 0..255 の表示輝度 (ウィンドウ処理)"""
        lo = self.wc - self.ww / 2
        img = np.clip((a.astype(np.float32) - lo) * (255.0 / self.ww), 0, 255).astype(np.uint8)
        return 255 - img if self.invert else img


# ======================================================================
# 画像ビュー (Axial / Coronal / Sagittal 共通)
# ======================================================================
def _letter(v) -> str:
    """患者座標(LPS)ベクトルが向いている方向の解剖学的ラベル"""
    i = int(np.argmax(np.abs(v)))
    return [("R", "L"), ("A", "P"), ("I", "S")][i][1 if v[i] > 0 else 0]


def _opposite(c: str) -> str:
    return {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}[c]


def _dg(ds, name, default=""):
    v = getattr(ds, name, None)
    return default if v is None or v == "" else v


def _private(ds, group, elem, default=""):
    el = ds.get((group, elem))
    if el is None:
        return default
    v = el.value
    if isinstance(v, bytes):
        v = v.decode("latin1").strip("\x00 ")
    return str(v)


PROJ_MODES = [("通常", "none"), ("MIP (最大値)", "mip"), ("MinIP (最小値)", "minip"), ("平均", "mean")]


def measure_angle(p1, p2, p3, sx, sy):
    """画像座標(px)の3点から、p2 を頂点とする角度[度]。画素間隔で mm に直して計算する。"""
    v1 = np.array([(p1[0] - p2[0]) * sx, (p1[1] - p2[1]) * sy])
    v2 = np.array([(p3[0] - p2[0]) * sx, (p3[1] - p2[1]) * sy])
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def roi_stats(a, p1, p2, sx, sy):
    """画像座標の外接矩形 p1-p2 を持つ楕円 ROI の CT値統計"""
    H, W = a.shape
    cx, cy = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    ax, ay = abs(p2[0] - p1[0]) / 2, abs(p2[1] - p1[1]) / 2
    if ax < 0.5 or ay < 0.5:
        return None
    x0, x1 = max(0, int(cx - ax)), min(W, int(cx + ax) + 2)
    y0, y1 = max(0, int(cy - ay)), min(H, int(cy + ay) + 2)
    if x0 >= x1 or y0 >= y1:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = ((xx + .5 - cx) / ax) ** 2 + ((yy + .5 - cy) / ay) ** 2 <= 1
    vals = a[y0:y1, x0:x1][mask].astype(np.float64)
    if vals.size == 0:
        return None
    return {"mean": vals.mean(), "std": vals.std(), "min": vals.min(), "max": vals.max(),
            "area": vals.size * sx * sy, "n": int(vals.size)}


def slab_project(vol, axis, idx, n, mode):
    """axis 方向に idx を中心とした n 枚のスラブを投影 (mode: none/mip/minip/mean)"""
    if mode == "none" or n <= 1:
        return vol.take(idx, axis=axis)
    lo = max(0, idx - n // 2)
    hi = min(vol.shape[axis], idx - n // 2 + n)
    sl = [slice(None)] * 3
    sl[axis] = slice(lo, hi)
    sub = vol[tuple(sl)]
    if mode == "mip":
        return sub.max(axis=axis)
    if mode == "minip":
        return sub.min(axis=axis)
    return sub.mean(axis=axis, dtype=np.float32)


def tissue_label(hu) -> str:
    """CT値から組織の目安 (あくまで概算)"""
    for limit, name in ((-900, "空気"), (-200, "肺野・空気混在"), (-20, "脂肪"), (15, "水・髄液"),
                        (70, "軟部組織(脳・筋)"), (300, "高吸収(出血・造影・石灰化)")):
        if hu < limit:
            return name
    return "骨"


def _text(p: QtGui.QPainter, x, y, s, color=QtGui.QColor(255, 235, 120)):
    p.setPen(QtGui.QColor(0, 0, 0, 220))
    p.drawText(int(x) + 1, int(y) + 1, s)
    p.setPen(color)
    p.drawText(int(x), int(y), s)


class ImageView(QtWidgets.QWidget):
    hovered = Signal(object, float, float)     # (view, col, row) / 画面外は (view, nan, nan)
    measured = Signal(str)

    TITLES = {"axial": "Axial", "coronal": "Coronal", "sagittal": "Sagittal"}

    def __init__(self, plane: str, state: State):
        super().__init__()
        self.plane, self.state = plane, state
        self.zoom, self.pan = 1.0, QPointF(0, 0)
        self.meas: dict[int, list] = {}
        self._qimg = None
        self._buf = None
        self._key = None
        self._drag = None
        self._last = QPointF()
        self._temp = None
        self._hu = None
        self._hu_key = None
        self._angle_pts: list = []
        self._angle_slice = None
        self._cursor = (0.0, 0.0)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        state.series_changed.connect(self._on_series)
        state.cross_changed.connect(self.update)
        state.window_changed.connect(self.update)
        state.proj_changed.connect(self.update)
        state.style_changed.connect(self._on_style)
        self._on_style()

    def sizeHint(self):
        return QtCore.QSize(420, 420)

    # ---- 断面の幾何 ----------------------------------------------------
    @property
    def ser(self) -> Series | None:
        return self.state.series

    def dims(self):
        nz, ny, nx = self.ser.shape
        return {"axial": (ny, nx), "coronal": (nz, nx), "sagittal": (nz, ny)}[self.plane]   # (H, W)

    def spacing(self):
        s = self.ser
        return {"axial": (s.sx, s.sy), "coronal": (s.sx, s.dz), "sagittal": (s.sy, s.dz)}[self.plane]

    def slice_index(self):
        z, y, x = self.state.cross
        return {"axial": z, "coronal": y, "sagittal": x}[self.plane]

    def slice_count(self):
        nz, ny, nx = self.ser.shape
        return {"axial": nz, "coronal": ny, "sagittal": nx}[self.plane]

    def slice_hu(self) -> np.ndarray:
        """表示中の断面のCT値 (スラブ投影が有効なら投影後)。同条件ならキャッシュを返す"""
        st, s, i = self.state, self.ser, self.slice_index()
        key = (id(s), self.plane, i, st.proj_mode, st.slab_mm)
        if key != self._hu_key:
            axis = {"axial": 0, "coronal": 1, "sagittal": 2}[self.plane]
            step = {"axial": s.dz, "coronal": s.sy, "sagittal": s.sx}[self.plane]
            a = slab_project(s.volume, axis, i, max(1, round(st.slab_mm / step)), st.proj_mode)
            self._hu = a if self.plane == "axial" else a[::-1]      # Coronal/Sagittal は頭側が上
            self._hu_key = key
        return self._hu

    def cross_pos(self):
        z, y, x = self.state.cross
        nz = self.ser.shape[0]
        return {"axial": (x + .5, y + .5), "coronal": (x + .5, nz - 1 - z + .5),
                "sagittal": (y + .5, nz - 1 - z + .5)}[self.plane]

    def voxel_at(self, col, row):
        """画像座標 -> (z, y, x)"""
        z0, y0, x0 = self.state.cross
        nz = self.ser.shape[0]
        c, r = int(math.floor(col)), int(math.floor(row))
        return {"axial": (z0, r, c), "coronal": (nz - 1 - r, y0, c), "sagittal": (nz - 1 - r, c, x0)}[self.plane]

    def set_cross_from_image(self, col, row):
        H, W = self.dims()
        col, row = min(max(col, 0), W - 1e-6), min(max(row, 0), H - 1e-6)
        self.state.set_cross(*self.voxel_at(col, row))

    def edge_labels(self):
        s = self.ser
        right = {"axial": s.row_dir, "coronal": s.row_dir, "sagittal": s.col_dir}[self.plane]
        bottom = {"axial": s.col_dir, "coronal": -s.normal, "sagittal": -s.normal}[self.plane]
        r, b = _letter(right), _letter(bottom)
        return _opposite(b), b, _opposite(r), r     # 上, 下, 左, 右

    # ---- 座標変換 ------------------------------------------------------
    def _k(self):
        H, W = self.dims()
        sx, sy = self.spacing()
        return min(self.width() / (W * sx), self.height() / (H * sy)) * 0.98 * self.zoom

    def _origin(self):
        H, W = self.dims()
        sx, sy = self.spacing()
        k = self._k()
        return QPointF(self.width() / 2 + self.pan.x() - W * sx * k / 2,
                       self.height() / 2 + self.pan.y() - H * sy * k / 2)

    def to_screen(self, c, r):
        o, k, (sx, sy) = self._origin(), self._k(), self.spacing()
        return QPointF(o.x() + c * sx * k, o.y() + r * sy * k)

    def to_image(self, p: QPointF):
        o, k, (sx, sy) = self._origin(), self._k(), self.spacing()
        return (p.x() - o.x()) / (sx * k), (p.y() - o.y()) / (sy * k)

    # ---- 状態変化 ------------------------------------------------------
    def reset_view(self):
        self.zoom, self.pan = 1.0, QPointF(0, 0)
        self.update()

    def _on_series(self):
        self.meas.clear()
        self._angle_pts = []
        self._hu_key = None
        self._key = None
        self.reset_view()

    def _on_style(self):
        cur = {"pan": Qt.OpenHandCursor, "window": Qt.SizeAllCursor}.get(self.state.tool, Qt.CrossCursor)
        self.setCursor(cur)
        if self.state.tool != "angle":
            self._angle_pts = []
        self.update()

    def clear_current(self):
        if self.ser:
            self.meas.pop(self.slice_index(), None)
            self.update()

    def clear_all(self):
        self.meas.clear()
        self.update()

    # ---- 計測 ----------------------------------------------------------
    def _make_item(self, kind, p1, p2):
        sx, sy = self.spacing()
        item = {"kind": kind, "p1": p1, "p2": p2, "lines": []}
        if kind == "dist":
            d = math.hypot((p2[0] - p1[0]) * sx, (p2[1] - p1[1]) * sy)
            item["value"] = d
            item["lines"] = [f"{d:.1f} mm"]
        else:
            st = self._roi_stats(p1, p2)
            item["stats"] = st
            if st:
                item["lines"] = [f"Mean {st['mean']:.1f} HU", f"SD {st['std']:.1f}",
                                 f"Min {st['min']:.0f} / Max {st['max']:.0f}", f"Area {st['area']:.1f} mm2"]
        return item

    def _roi_stats(self, p1, p2):
        return roi_stats(self.slice_hu(), p1, p2, *self.spacing())

    def _make_angle(self, pts):
        item = {"kind": "angle", "pts": list(pts), "lines": []}
        if len(pts) == 3:
            a = measure_angle(*pts, *self.spacing())
            if a is not None:
                item["value"] = a
                item["lines"] = [f"{a:.1f} deg"]
        return item

    def _finish_angle(self):
        item = self._make_angle(self._angle_pts)
        self._angle_pts = []
        if "value" in item:
            idx = self.slice_index()
            self.meas.setdefault(idx, []).append(item)
            self.measured.emit(f"[{self.TITLES[self.plane]} {idx + 1}/{self.slice_count()}] 角度: {item['value']:.1f} deg")

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape and self._angle_pts:
            self._angle_pts = []                     # 作図中の角度を取り消す
            self.update()
        else:
            super().keyPressEvent(e)

    def _finish_item(self):
        item, self._temp = self._temp, None
        if not item:
            return
        if item["kind"] == "dist" and item["value"] < 0.5:
            return
        if item["kind"] == "roi" and not item.get("stats"):
            return
        idx = self.slice_index()
        self.meas.setdefault(idx, []).append(item)
        title = f"[{self.TITLES[self.plane]} {idx + 1}/{self.slice_count()}]"
        if item["kind"] == "dist":
            self.measured.emit(f"{title} 距離: {item['value']:.2f} mm")
        else:
            st = item["stats"]
            self.measured.emit(f"{title} ROI: Mean {st['mean']:.1f} HU, SD {st['std']:.1f}, "
                               f"Min {st['min']:.0f}, Max {st['max']:.0f}, Area {st['area']:.1f} mm2 ({st['n']} px)")

    # ---- 描画 ----------------------------------------------------------
    def _image(self):
        key = (id(self.ser), self.plane, self.slice_index(), self.state.wc, self.state.ww, self.state.invert,
               self.state.proj_mode, self.state.slab_mm)
        if key != self._key:
            img = np.ascontiguousarray(self.state.to_display(self.slice_hu()))
            H, W = img.shape
            self._buf = img
            self._qimg = QtGui.QImage(img.data, W, H, W, QtGui.QImage.Format_Grayscale8)
            self._key = key
        return self._qimg

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0))
        if self.ser is None:
            p.setPen(QtGui.QColor(160, 160, 160))
            p.drawText(self.rect(), Qt.AlignCenter, "フォルダを開いてください (Ctrl+O)")
            return
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        sx, sy = self.spacing()
        k, o = self._k(), self._origin()
        p.save()
        p.translate(o)
        p.scale(sx * k, sy * k)
        p.drawImage(0, 0, self._image())
        p.restore()
        if self.state.show_cross:
            c = self.to_screen(*self.cross_pos())
            p.setPen(QtGui.QPen(QtGui.QColor(60, 255, 60, 140), 1))
            p.drawLine(QPointF(c.x(), 0), QPointF(c.x(), self.height()))
            p.drawLine(QPointF(0, c.y()), QPointF(self.width(), c.y()))
        self._paint_measurements(p)
        self._paint_overlay(p)
        p.setPen(QtGui.QColor(70, 70, 70))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def _paint_measurements(self, p):
        idx = self.slice_index()
        items = list(self.meas.get(idx, []))
        if self._temp:
            items.append(self._temp)
        if self._angle_pts and self._angle_slice != idx:
            self._angle_pts = []                     # スライスが変わったら作図中の角度は破棄
        if self._angle_pts:
            items.append(self._make_angle(self._angle_pts + ([self._cursor] if len(self._angle_pts) < 3 else [])))
        f = p.font()
        f.setPointSize(9)
        p.setFont(f)
        cy = QtGui.QColor(0, 220, 255)
        fm = p.fontMetrics()
        for it in items:
            p.setPen(QtGui.QPen(cy, 1.6))
            if it["kind"] == "angle":
                pts = [self.to_screen(*q) for q in it["pts"]]
                p.drawPolyline(pts)
                for q in pts:
                    p.drawEllipse(q, 3, 3)
                if len(pts) < 3:
                    continue
                tx, ty = pts[1].x() + 8, pts[1].y() - 8
            elif it["kind"] == "dist":
                a, b = self.to_screen(*it["p1"]), self.to_screen(*it["p2"])
                p.drawLine(a, b)
                for q in (a, b):
                    p.drawEllipse(q, 3, 3)
                tx, ty = (a.x() + b.x()) / 2 + 6, (a.y() + b.y()) / 2 - 6
            else:
                a, b = self.to_screen(*it["p1"]), self.to_screen(*it["p2"])
                r = QRectF(a, b).normalized()
                p.setBrush(QtGui.QColor(0, 220, 255, 30))
                p.drawEllipse(r)
                p.setBrush(Qt.NoBrush)
                tx, ty = r.right() + 6, r.top() + 12
            for i, line in enumerate(it["lines"]):
                _text(p, tx, ty + i * fm.height(), line, cy)

    def _paint_overlay(self, p):
        s, ds = self.ser, self.ser.datasets[0]
        z, y, x = self.state.cross
        nz, ny, nx = s.shape
        f = QtGui.QFont("Consolas", 9)
        f.setStyleHint(QtGui.QFont.Monospace)
        p.setFont(f)
        fm = p.fontMetrics()
        lh, w, h = fm.height(), self.width(), self.height()

        tl = [f"{self.TITLES[self.plane]}"]
        if self.state.show_patient:
            tl += [f"{_dg(ds, 'PatientName')}  ID:{_dg(ds, 'PatientID')}",
                   f"{_dg(ds, 'PatientSex')} {_dg(ds, 'PatientAge')}  DOB:{_dg(ds, 'PatientBirthDate')}"]
        sd = str(_dg(ds, "StudyDate"))
        if len(sd) == 8:
            sd = f"{sd[:4]}-{sd[4:6]}-{sd[6:]}"
        tl += [f"{_dg(ds, 'Modality')}  {_dg(ds, 'BodyPartExamined', 'HEAD')}  {sd}",
               f"{str(_dg(ds, 'Manufacturer')).strip()} {_dg(ds, 'ManufacturerModelName')}",
               f"{_dg(ds, 'InstitutionName')}"]

        tr = [f"{_dg(ds, 'KVP')} kV  {_dg(ds, 'XRayTubeCurrent')} mA",
              f"Kernel {_dg(ds, 'ConvolutionKernel')}  {_private(ds, 0x7005, 0x100b)}",
              f"Thick {_dg(ds, 'SliceThickness')} mm  Space {s.dz:.2f} mm",
              f"Pixel {s.sx:.3f} x {s.sy:.3f} mm",
              f"Recon FOV {_dg(ds, 'ReconstructionDiameter')} mm  {_dg(ds, 'PatientPosition')}"]

        idx = self.slice_index()
        if self.plane == "axial":
            d = s.datasets[z]
            bl = [f"Im {z + 1}/{nz}  (Inst {_dg(d, 'InstanceNumber')})",
                  f"Loc z = {s.ipps[z][2]:.2f} mm"]
        else:
            pt = s.patient_point(z, y, x)
            axis, val = ("y", pt[1]) if self.plane == "coronal" else ("x", pt[0])
            bl = [f"{'Cor' if self.plane == 'coronal' else 'Sag'} {idx + 1}/{self.slice_count()}",
                  f"Loc {axis} = {val:.2f} mm"]
        br = [f"W: {self.state.ww:.0f}  L: {self.state.wc:.0f}", f"Zoom {self.zoom * 100:.0f}%"]
        if self.state.proj_mode != "none":
            name = {v: k for k, v in PROJ_MODES}[self.state.proj_mode].split(" ")[0]
            br.insert(0, f"{name} {self.state.slab_mm:.0f} mm")

        for i, t in enumerate(tl):
            _text(p, 8, 8 + fm.ascent() + i * lh, t)
        for i, t in enumerate(tr):
            _text(p, w - 8 - fm.horizontalAdvance(t), 8 + fm.ascent() + i * lh, t)
        for i, t in enumerate(bl):
            _text(p, 8, h - 8 - fm.descent() - (len(bl) - 1 - i) * lh, t)
        for i, t in enumerate(br):
            _text(p, w - 8 - fm.horizontalAdvance(t), h - 8 - fm.descent() - (len(br) - 1 - i) * lh, t)

        # 向きラベル
        f2 = QtGui.QFont("Consolas", 12, QtGui.QFont.Bold)
        p.setFont(f2)
        fm2 = p.fontMetrics()
        top, bot, left, right = self.edge_labels()
        cw = fm2.horizontalAdvance("W")
        _text(p, w / 2 - cw / 2, 8 + fm2.ascent(), top, QtGui.QColor(120, 200, 255))
        _text(p, w / 2 - cw / 2, h - 10, bot, QtGui.QColor(120, 200, 255))
        _text(p, 8 + 0, h / 2, left, QtGui.QColor(120, 200, 255))
        _text(p, w - 10 - cw, h / 2, right, QtGui.QColor(120, 200, 255))

        # スケールバー
        p.setFont(f)
        ppm = self.spacing()[0] * self._k()
        cand = [v for v in (1, 2, 5, 10, 20, 50, 100, 200) if v * ppm <= w * 0.25]
        if cand:
            mm = cand[-1]
            L = mm * ppm
            x0, y0 = w / 2 - L / 2, h - 26
            p.setPen(QtGui.QPen(QtGui.QColor(255, 235, 120), 2))
            p.drawLine(QPointF(x0, y0), QPointF(x0 + L, y0))
            p.drawLine(QPointF(x0, y0 - 4), QPointF(x0, y0 + 4))
            p.drawLine(QPointF(x0 + L, y0 - 4), QPointF(x0 + L, y0 + 4))
            label = f"{mm} mm"
            _text(p, w / 2 - fm.horizontalAdvance(label) / 2, y0 - 6, label)

    # ---- マウス操作 ----------------------------------------------------
    def mousePressEvent(self, e):
        if self.ser is None:
            return
        self.setFocus()
        self._last = e.position()
        col, row = self.to_image(e.position())
        tool = self.state.tool
        if e.button() == Qt.RightButton:
            self._drag = "window"
        elif e.button() == Qt.MiddleButton:
            self._drag = "pan"
        elif e.button() == Qt.LeftButton:
            self._drag = tool
            if tool == "cross":
                self.set_cross_from_image(col, row)
            elif tool in ("dist", "roi"):
                self._temp = self._make_item(tool, (col, row), (col, row))
            elif tool == "angle":
                self._angle_slice = self.slice_index()
                self._angle_pts.append((col, row))
                if len(self._angle_pts) == 3:
                    self._finish_angle()
                self.update()
        if self._drag == "pan":
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self.ser is None:
            return
        col, row = self.to_image(e.position())
        self._cursor = (col, row)
        self.hovered.emit(self, col, row)
        if self._angle_pts:
            self.update()
        d = e.position() - self._last
        self._last = e.position()
        if self._drag == "cross":
            self.set_cross_from_image(col, row)
        elif self._drag == "pan":
            self.pan += d
            self.update()
        elif self._drag == "window":
            f = max(self.state.ww, 50) / 200
            self.state.set_window(self.state.wc + d.y() * f, self.state.ww + d.x() * f)
        elif self._drag in ("dist", "roi") and self._temp:
            self._temp = self._make_item(self._drag, self._temp["p1"], (col, row))
            self.update()

    def mouseReleaseEvent(self, e):
        if self._drag in ("dist", "roi"):
            self._finish_item()
        self._drag = None
        self._on_style()

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton and self.state.tool in ("pan", "window"):
            self.reset_view()

    def leaveEvent(self, e):
        self.hovered.emit(self, float("nan"), float("nan"))

    def wheelEvent(self, e):
        if self.ser is None:
            return
        dy = e.angleDelta().y()
        if dy == 0:
            return
        if e.modifiers() & Qt.ControlModifier:
            before = self.to_image(e.position())
            self.zoom = min(max(self.zoom * (1.15 if dy > 0 else 1 / 1.15), 0.2), 40)
            self.pan += e.position() - self.to_screen(*before)      # カーソル位置を固定してズーム
            self.update()
        else:
            self.step((5 if e.modifiers() & Qt.ShiftModifier else 1) * (-1 if dy > 0 else 1))

    def step(self, delta: int):
        """この断面のスライス位置を delta 枚動かす"""
        if self.ser is None:
            return
        z, y, x = self.state.cross
        if self.plane == "axial":
            self.state.set_cross(z + delta, y, x)
        elif self.plane == "coronal":
            self.state.set_cross(z, y + delta, x)
        else:
            self.state.set_cross(z, y, x + delta)


# ======================================================================
# DICOM タグ一覧
# ======================================================================
class TagDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DICOM タグ一覧")
        self.resize(900, 700)
        lay = QtWidgets.QVBoxLayout(self)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("検索 (タグ番号・名称・値)  例: 0028 / Pixel / CT")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)
        self.title = QtWidgets.QLabel()
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Tag", "名称", "VR", "値"])
        self.tree.setAlternatingRowColors(True)
        for i, w in enumerate((110, 280, 40)):
            self.tree.setColumnWidth(i, w)
        lay.addWidget(self.title)
        lay.addWidget(self.search)
        lay.addWidget(self.tree)

    @staticmethod
    def _value_str(el) -> str:
        v = el.value
        if isinstance(v, bytes):
            if all(32 <= b < 127 or b == 0 for b in v):
                s = v.decode("ascii").strip("\x00 ")
                return s
            hexs = " ".join(f"{b:02X}" for b in v[:16]) + (" ..." if len(v) > 16 else "")
            return f"{hexs}  ({len(v)} bytes)"
        s = str(v).replace("\n", " ")
        return s if len(s) <= 300 else s[:300] + " ..."

    def _add(self, parent, ds):
        for el in ds:
            tag = f"({el.tag.group:04X},{el.tag.element:04X})"
            name = el.name if not el.tag.is_private else f"[Private] {el.name}"
            it = QtWidgets.QTreeWidgetItem(parent, [tag, name, str(el.VR), "" if el.VR == "SQ" else self._value_str(el)])
            if el.VR == "SQ":
                for n, sub in enumerate(el.value):
                    sit = QtWidgets.QTreeWidgetItem(it, ["", f"Item {n + 1}", "", ""])
                    self._add(sit, sub)

    def set_dataset(self, ds, title: str):
        self.title.setText(title)
        self.tree.clear()
        if getattr(ds, "file_meta", None):
            meta = QtWidgets.QTreeWidgetItem(self.tree, ["", "File Meta Information", "", ""])
            self._add(meta, ds.file_meta)
            meta.setExpanded(True)
        self._add(self.tree.invisibleRootItem(), ds)
        self._apply_filter(self.search.text())

    def _apply_filter(self, text: str):
        text = text.strip().lower()

        def visit(it) -> bool:
            child_vis = False
            for i in range(it.childCount()):
                child_vis |= visit(it.child(i))
            own = any(text in it.text(c).lower() for c in range(4))
            it.setHidden(bool(text) and not (own or child_vis))
            if text and child_vis:
                it.setExpanded(True)
            return own or child_vis or not text

        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            visit(root.child(i))


# ======================================================================
# メインウィンドウ
# ======================================================================
class HistogramWidget(QtWidgets.QWidget):
    """現在の Axial スライスの CT値ヒストグラム (対数)。窓範囲を重ね、ドラッグで窓を指定できる"""
    LO, HI, BIN = -1100, 2000, 20

    def __init__(self, state: State):
        super().__init__()
        self.state = state
        self.setFixedHeight(150)
        self._key = None
        self._counts = None
        self._x0 = None
        state.cross_changed.connect(self.update)
        state.window_changed.connect(self.update)
        state.series_changed.connect(self.update)

    def _hist(self):
        s, z = self.state.series, self.state.cross[0]
        if (id(s), z) != self._key:
            self._counts, _ = np.histogram(s.volume[z], bins=np.arange(self.LO, self.HI + self.BIN, self.BIN))
            self._key = (id(s), z)
        return self._counts

    def _plot_rect(self):
        return QRectF(10, 18, self.width() - 20, self.height() - 18 - 18)

    def _x_of(self, hu):
        r = self._plot_rect()
        return r.left() + (hu - self.LO) / (self.HI - self.LO) * r.width()

    def _hu_of(self, x):
        r = self._plot_rect()
        return self.LO + (x - r.left()) / r.width() * (self.HI - self.LO)

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(30, 30, 32))
        p.setFont(QtGui.QFont("Consolas", 8))
        if self.state.series is None:
            return
        r = self._plot_rect()
        c = self._hist()
        h = np.log10(1 + c.astype(np.float64))
        h = h / max(h.max(), 1e-9)
        bw = r.width() / len(c)
        p.setPen(Qt.NoPen)
        p.setBrush(QtGui.QColor(120, 160, 220))
        for i, v in enumerate(h):
            if v > 0:
                p.drawRect(QRectF(r.left() + i * bw, r.bottom() - v * r.height(), max(bw - 0.5, 0.5), v * r.height()))
        lo, hi = self.state.wc - self.state.ww / 2, self.state.wc + self.state.ww / 2
        xl, xh = self._x_of(max(lo, self.LO)), self._x_of(min(hi, self.HI))
        p.setBrush(QtGui.QColor(255, 220, 60, 55))
        p.setPen(QtGui.QPen(QtGui.QColor(255, 220, 60), 1))
        p.drawRect(QRectF(xl, r.top(), max(xh - xl, 1), r.height()))
        p.setPen(QtGui.QColor(160, 160, 160))
        for t in range(-1000, 2001, 500):
            x = self._x_of(t)
            p.drawLine(QPointF(x, r.bottom()), QPointF(x, r.bottom() + 3))
            p.drawText(QRectF(x - 25, r.bottom() + 3, 50, 12), Qt.AlignCenter, str(t))
        p.setPen(QtGui.QColor(255, 235, 120))
        p.drawText(QRectF(10, 2, self.width() - 20, 14), Qt.AlignLeft,
                   f"CT値ヒストグラム (Axial {self.state.cross[0] + 1}, 対数)   窓: {lo:.0f} ~ {hi:.0f} HU   [ドラッグで窓指定]")

    def mousePressEvent(self, e):
        if self.state.series is not None and e.button() == Qt.LeftButton:
            self._x0 = self._hu_of(e.position().x())

    def mouseMoveEvent(self, e):
        if self._x0 is not None:
            x1 = self._hu_of(e.position().x())
            if abs(x1 - self._x0) >= 1:
                self.state.set_window((self._x0 + x1) / 2, abs(x1 - self._x0))

    def mouseReleaseEvent(self, e):
        self._x0 = None


HELP_TEXT = """\
【操作】
 ホイール       : スライス送り (Shiftで5枚ずつ)
 Ctrl+ホイール  : ズーム (カーソル位置基準)
 右ドラッグ     : ウィンドウ調整 (横=幅 W / 縦=レベル L)
 中ドラッグ     : パン
 左クリック/ドラッグ: ツールに従う
   C 位置(十字線)  W ウィンドウ  H パン
   D 距離計測      E 楕円ROI (Mean/SD/Min/Max/面積)
   A 角度計測 (3点クリック、頂点は2点目 / Escで取消)
 1-6 プリセット  I 白黒反転  X 十字線  R ズーム解除
 M 投影切替 (通常/MIP/MinIP/平均、厚みはツールバー)
 Space シネ再生  矢印/PageUp/PageDown/Home/End スライス送り
 ヒストグラム上をドラッグ: 窓範囲を指定
 P 患者情報表示  T タグ一覧  L Axialのみ/MPR切替
 Del 現スライスの計測消去 / Ctrl+Del 全消去
 Ctrl+O フォルダを開く / Ctrl+S 画面を保存
"""


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Viewer")
        self.resize(1400, 900)
        self.state = State()
        self.all_series: list[Series] = []
        self.views = {p: ImageView(p, self.state) for p in ("axial", "coronal", "sagittal")}
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QtGui.QFont("Consolas", 9))
        self.tag_dialog: TagDialog | None = None
        self._tag_z = None
        self.active_view = self.views["axial"]      # 矢印キーの操作対象 (最後に触れたビュー)
        self.hist = HistogramWidget(self.state)
        self.side = QtWidgets.QWidget()
        side_lay = QtWidgets.QVBoxLayout(self.side)
        side_lay.setContentsMargins(0, 0, 0, 0)
        side_lay.setSpacing(2)
        side_lay.addWidget(self.hist)
        side_lay.addWidget(self.log)
        self.cine_timer = QtCore.QTimer(self)
        self.cine_timer.timeout.connect(self._cine_tick)

        self.grid = grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setSpacing(2)
        grid.addWidget(self.views["axial"], 0, 0)
        grid.addWidget(self.views["coronal"], 0, 1)
        grid.addWidget(self.views["sagittal"], 1, 0)
        grid.addWidget(self.side, 1, 1)
        for c in (0, 1):
            grid.setColumnStretch(c, 1)
            grid.setRowStretch(c, 1)
        central = QtWidgets.QWidget()
        central.setLayout(grid)
        self.setCentralWidget(central)

        self.hover_label = QtWidgets.QLabel("")
        self.hover_label.setFont(QtGui.QFont("Consolas", 9))
        self.statusBar().addWidget(self.hover_label, 1)

        self._build_actions()
        for v in self.views.values():
            v.hovered.connect(self._on_hover)
            v.hovered.connect(lambda view, *_: setattr(self, "active_view", view))
            v.measured.connect(self.log.appendPlainText)
        self.state.window_changed.connect(self._sync_window_widgets)
        self.state.cross_changed.connect(self._refresh_tags)

    # ---- UI 構築 -------------------------------------------------------
    def _act(self, text, key=None, slot=None, checkable=False, checked=False):
        a = QtGui.QAction(text, self)
        if key:
            a.setShortcut(key)
        a.setCheckable(checkable)
        if checkable:
            a.setChecked(checked)
        if slot:
            (a.toggled if checkable else a.triggered).connect(slot)
        return a

    def _build_actions(self):
        tb = self.addToolBar("main")
        tb.setMovable(False)
        m_file, m_view, m_tool = self.menuBar().addMenu("ファイル"), self.menuBar().addMenu("表示"), self.menuBar().addMenu("ツール")

        open_a = self._act("フォルダを開く...", "Ctrl+O", self.open_folder_dialog)
        save_a = self._act("画面を保存...", "Ctrl+S", self.save_screenshot)
        m_file.addActions([open_a, save_a])
        tb.addAction(open_a)
        tb.addAction(save_a)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel(" シリーズ: "))
        self.series_combo = QtWidgets.QComboBox()
        self.series_combo.setMinimumWidth(260)
        self.series_combo.currentIndexChanged.connect(self._on_series_selected)
        tb.addWidget(self.series_combo)
        tb.addSeparator()

        group = QtGui.QActionGroup(self)
        for name, key, tool in (("位置(C)", "C", "cross"), ("窓(W)", "W", "window"), ("パン(H)", "H", "pan"),
                                ("距離(D)", "D", "dist"), ("ROI(E)", "E", "roi"), ("角度(A)", "A", "angle")):
            a = self._act(name, key, lambda on, t=tool: on and self._set_tool(t), True, tool == "cross")
            group.addAction(a)
            tb.addAction(a)
            m_tool.addAction(a)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel(" プリセット: "))
        self.preset_combo = QtWidgets.QComboBox()
        for i, (n, wc, ww) in enumerate(PRESETS):
            self.preset_combo.addItem(n if wc is None else f"{n} ({wc}/{ww})")
        self.preset_combo.activated.connect(self._apply_preset)
        tb.addWidget(self.preset_combo)
        for i in range(len(PRESETS)):
            a = self._act(f"プリセット{i + 1}", str(i + 1), lambda _=False, i=i: self._apply_preset(i))
            self.addAction(a)
        self.wc_spin = QtWidgets.QDoubleSpinBox()
        self.ww_spin = QtWidgets.QDoubleSpinBox()
        for sp, lo, hi in ((self.wc_spin, -5000, 10000), (self.ww_spin, 1, 20000)):
            sp.setRange(lo, hi)
            sp.setDecimals(0)
            sp.setFixedWidth(80)
            sp.valueChanged.connect(self._on_spin)
        tb.addWidget(QtWidgets.QLabel(" L:"))
        tb.addWidget(self.wc_spin)
        tb.addWidget(QtWidgets.QLabel(" W:"))
        tb.addWidget(self.ww_spin)

        self.addToolBarBreak()
        tb2 = self.addToolBar("view")
        tb2.setMovable(False)
        tb2.addWidget(QtWidgets.QLabel(" 投影: "))
        self.proj_combo = QtWidgets.QComboBox()
        for name, _ in PROJ_MODES:
            self.proj_combo.addItem(name)
        self.proj_combo.currentIndexChanged.connect(self._on_proj)
        tb2.addWidget(self.proj_combo)
        self.slab_spin = QtWidgets.QDoubleSpinBox()
        self.slab_spin.setRange(1, 200)
        self.slab_spin.setDecimals(0)
        self.slab_spin.setValue(self.state.slab_mm)
        self.slab_spin.setSuffix(" mm")
        self.slab_spin.valueChanged.connect(self._on_proj)
        tb2.addWidget(QtWidgets.QLabel(" 厚み:"))
        tb2.addWidget(self.slab_spin)
        self.addAction(self._act("投影切替", "M", lambda: self.proj_combo.setCurrentIndex(
            (self.proj_combo.currentIndex() + 1) % len(PROJ_MODES))))
        tb2.addSeparator()

        self.cine_act = self._act("再生(Space)", "Space", self._toggle_cine, True)
        tb2.addAction(self.cine_act)
        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.valueChanged.connect(lambda v: self.cine_timer.setInterval(int(1000 / v)))
        tb2.addWidget(self.fps_spin)
        for key, d in (("Up", -1), ("Left", -1), ("Down", 1), ("Right", 1), ("PgUp", -10), ("PgDown", 10)):
            self.addAction(self._act(f"スライス{d:+d}", key, lambda _=False, d=d: self.active_view.step(d)))
        self.addAction(self._act("先頭", "Home", lambda: self.active_view.step(-10 ** 6)))
        self.addAction(self._act("末尾", "End", lambda: self.active_view.step(10 ** 6)))
        tb2.addSeparator()

        inv = self._act("反転(I)", "I", self._set_invert, True)
        crs = self._act("十字線(X)", "X", self._set_show_cross, True, True)
        pat = self._act("患者情報(P)", "P", self._set_show_patient, True, False)
        lay = self._act("Axialのみ(L)", "L", self._set_axial_only, True, False)
        rst = self._act("ズーム解除(R)", "R", lambda: [v.reset_view() for v in self.views.values()])
        tags = self._act("タグ一覧(T)", "T", self.show_tags)
        clr = self._act("計測消去(Del)", "Del", lambda: [v.clear_current() for v in self.views.values()])
        clr_all = self._act("全計測消去", "Ctrl+Del", lambda: [v.clear_all() for v in self.views.values()])
        for a in (inv, crs, pat, lay, rst, tags, clr, clr_all):
            m_view.addAction(a)
        for a in (inv, crs, pat, lay, tags, clr):
            tb2.addAction(a)

    # ---- 読み込み ------------------------------------------------------
    def load_folder(self, folder: str):
        dlg = QtWidgets.QProgressDialog("DICOM を読み込み中...", "中止", 0, 100, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)

        def progress(i, n):
            dlg.setMaximum(n)
            dlg.setValue(i)
            QtWidgets.QApplication.processEvents()
            return not dlg.wasCanceled()

        try:
            series = read_folder(folder, progress)
        finally:
            dlg.close()
        if not series:
            QtWidgets.QMessageBox.warning(self, "読み込み", "表示できる DICOM シリーズが見つかりませんでした。")
            return
        self.all_series = series
        self.series_combo.blockSignals(True)
        self.series_combo.clear()
        for s in series:
            self.series_combo.addItem(f"#{s.number} {s.description} ({s.shape[0]}枚)")
        self.series_combo.blockSignals(False)
        self._on_series_selected(0)
        self.statusBar().showMessage(f"{folder}: {len(series)} シリーズ読み込み", 5000)

    def open_folder_dialog(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "DICOM フォルダを選択")
        if d:
            self.load_folder(d)

    def _on_series_selected(self, i):
        if 0 <= i < len(self.all_series):
            s = self.all_series[i]
            self.state.set_series(s)
            self._refresh_tags()
            self._write_summary(s)

    def _write_summary(self, s: Series):
        ds = s.datasets[0]
        nz, ny, nx = s.shape
        lines = [
            f"シリーズ #{s.number}: {s.description}",
            f"UID: {s.uid}",
            f"枚数 {nz} / 画像 {nx} x {ny} px / {ds.BitsStored}bit "
            f"({'符号付き' if ds.PixelRepresentation else '符号なし'})",
            f"画素間隔 {s.sx:.4f} x {s.sy:.4f} mm / スライス間隔 {s.dz:.3f} mm / 厚み {_dg(ds, 'SliceThickness')} mm",
            f"位置範囲 {s.positions[0]:.1f} ~ {s.positions[-1]:.1f} mm  体位 {_dg(ds, 'PatientPosition')}",
            f"CT値範囲 {int(s.volume.min())} ~ {int(s.volume.max())} HU  既定 W/L = {s.default_window[1]:.0f}/{s.default_window[0]:.0f}",
        ] + [f"! {w}" for w in s.warnings] + ["", HELP_TEXT, "---- 計測結果 ----"]
        self.log.setPlainText("\n".join(lines))

    # ---- 状態同期 ------------------------------------------------------
    def _set_tool(self, t):
        self.state.tool = t
        self.state.style_changed.emit()

    def _set_invert(self, on):
        self.state.invert = on
        self.state.window_changed.emit()

    def _set_show_cross(self, on):
        self.state.show_cross = on
        self.state.style_changed.emit()

    def _set_show_patient(self, on):
        self.state.show_patient = on
        self.state.style_changed.emit()

    def _set_axial_only(self, on):
        for w in (self.views["coronal"], self.views["sagittal"], self.side):
            w.setVisible(not on)
        self.grid.setColumnStretch(1, 0 if on else 1)      # 非表示の列・行にスペースを残さない
        self.grid.setRowStretch(1, 0 if on else 1)

    def _apply_preset(self, i):
        if self.state.series is None:
            return
        _, wc, ww = PRESETS[i]
        if wc is None:
            wc, ww = self.state.series.default_window
        self.preset_combo.setCurrentIndex(i)
        self.state.set_window(wc, ww)

    def _on_proj(self, *_):
        self.state.set_projection(PROJ_MODES[self.proj_combo.currentIndex()][1], self.slab_spin.value())

    def _toggle_cine(self, on):
        if on and self.state.series is not None:
            self.cine_timer.start(int(1000 / self.fps_spin.value()))
        else:
            self.cine_timer.stop()
            self.cine_act.setChecked(False)

    def _cine_tick(self):
        z, y, x = self.state.cross
        nz = self.state.series.shape[0]
        self.state.set_cross((z + 1) % nz, y, x)        # Axial を先頭スライスからループ再生

    def _on_spin(self):
        if self.state.series is not None:
            self.state.set_window(self.wc_spin.value(), self.ww_spin.value())

    def _sync_window_widgets(self):
        for sp, v in ((self.wc_spin, self.state.wc), (self.ww_spin, self.state.ww)):
            sp.blockSignals(True)
            sp.setValue(round(v))
            sp.blockSignals(False)

    def _on_hover(self, view: ImageView, col, row):
        if math.isnan(col):
            self.hover_label.setText("")
            return
        H, W = view.dims()
        if not (0 <= col < W and 0 <= row < H):
            self.hover_label.setText("")
            return
        z, y, x = view.voxel_at(col, row)
        s = self.state.series
        pt = s.patient_point(z, y, x)
        hu = float(view.slice_hu()[int(row), int(col)])        # 投影表示中はその値
        self.hover_label.setText(
            f"{view.TITLES[view.plane]}  voxel(x,y,z)=({x},{y},{z})  HU={hu:.0f} ({tissue_label(hu)})  "
            f"患者座標(mm) X={pt[0]:.1f} Y={pt[1]:.1f} Z={pt[2]:.1f}")

    # ---- タグ一覧・保存 ------------------------------------------------
    def show_tags(self):
        if self.state.series is None:
            return
        if self.tag_dialog is None:
            self.tag_dialog = TagDialog(self)
        self._tag_z = None
        self.tag_dialog.show()
        self.tag_dialog.raise_()
        self._refresh_tags()

    def _refresh_tags(self):
        if self.tag_dialog is None or not self.tag_dialog.isVisible() or self.state.series is None:
            return
        z = self.state.cross[0]
        if z != self._tag_z:
            self._tag_z = z
            self.tag_dialog.set_dataset(self.state.series.datasets[z], f"Axial スライス {z + 1}/{self.state.series.shape[0]} のタグ (画素データは省略)")

    def save_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "画面を保存", "screenshot.png", "PNG (*.png)")
        if path:
            self.grab().save(path)


def apply_dark_theme(app):
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, c in ((QtGui.QPalette.Window, (45, 45, 48)), (QtGui.QPalette.WindowText, (230, 230, 230)),
                    (QtGui.QPalette.Base, (30, 30, 32)), (QtGui.QPalette.AlternateBase, (40, 40, 44)),
                    (QtGui.QPalette.Text, (230, 230, 230)), (QtGui.QPalette.Button, (60, 60, 64)),
                    (QtGui.QPalette.ButtonText, (230, 230, 230)), (QtGui.QPalette.Highlight, (0, 120, 200)),
                    (QtGui.QPalette.HighlightedText, (255, 255, 255))):
        pal.setColor(role, QtGui.QColor(*c))
    app.setPalette(pal)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication(argv)
    apply_dark_theme(app)
    win = MainWindow()
    win.show()
    folder = argv[1] if len(argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "DICOM", "DICOM")
    if os.path.isdir(folder):
        QtCore.QTimer.singleShot(0, lambda: win.load_folder(folder))
    else:
        QtCore.QTimer.singleShot(0, win.open_folder_dialog)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
