"""コア処理の単体テスト: python -m unittest discover -s tests"""
import math
import os
import sys
import unittest

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dicom_viewer as dv  # noqa: E402


def make_slice(z, instance, pixels, slope=1, intercept=0, signed=1, uid="1.2.3"):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SeriesInstanceUID = uid
    ds.SOPInstanceUID = generate_uid()
    ds.InstanceNumber = instance
    ds.ImagePositionPatient = [-10.0, -10.0, z]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [0.5, 0.25]
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = signed
    ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
    ds.PixelData = pixels.astype("<i2" if signed else "<u2").tobytes()
    return ds


class LoadingTest(unittest.TestCase):
    def test_sorted_by_position_not_instance_number(self):
        zs = [5.0, 1.0, 3.0, 0.0, 2.0, 4.0]
        # InstanceNumber を位置と無関係にして、位置順で並ぶことを確認
        dss = [make_slice(z, 100 - i, np.full((4, 4), int(z))) for i, z in enumerate(zs)]
        s = dv.build_series(dss)
        self.assertEqual(list(s.positions), [0, 1, 2, 3, 4, 5])
        self.assertEqual([int(v[0, 0]) for v in s.volume], [0, 1, 2, 3, 4, 5])
        self.assertAlmostEqual(s.dz, 1.0)
        self.assertEqual((s.sx, s.sy), (0.25, 0.5))      # PixelSpacing = [行, 列]

    def test_hu_conversion_unsigned(self):
        px = np.array([[0, 1024], [1500, 3000]])
        s = dv.build_series([make_slice(0, 1, px, slope=1, intercept=-1024, signed=0),
                             make_slice(1, 2, px, slope=1, intercept=-1024, signed=0)])
        self.assertEqual(s.volume[0].tolist(), [[-1024, 0], [476, 1976]])

    def test_hu_conversion_signed_and_slope(self):
        px = np.array([[-100, 0], [50, 100]])
        s = dv.build_series([make_slice(0, 1, px, slope=2, intercept=10),
                             make_slice(1, 2, px, slope=2, intercept=10)])
        self.assertEqual(s.volume[0].tolist(), [[-190, 10], [110, 210]])

    def test_default_window_fallback(self):
        px = np.arange(16).reshape(4, 4) * 10
        s = dv.build_series([make_slice(0, 1, px), make_slice(1, 2, px)])
        self.assertGreater(s.default_window[1], 0)


class DisplayTest(unittest.TestCase):
    def test_windowing(self):
        st = dv.State()
        st.wc, st.ww = 40, 80                    # 0 ~ 80 HU
        out = st.to_display(np.array([-500, 0, 40, 80, 500]))
        self.assertEqual(out[0], 0)
        self.assertEqual(out[1], 0)
        self.assertEqual(out[2], 127)
        self.assertEqual(out[3], 255)
        self.assertEqual(out[4], 255)
        st.invert = True
        self.assertEqual(st.to_display(np.array([0]))[0], 255)


class MeasureTest(unittest.TestCase):
    def test_angle(self):
        self.assertAlmostEqual(dv.measure_angle((1, 0), (0, 0), (0, 1), 1, 1), 90)
        self.assertAlmostEqual(dv.measure_angle((1, 0), (0, 0), (-1, 0), 1, 1), 180)
        # 画素間隔が異方性でも mm で計算される (x: 2mm/px, y: 1mm/px → 45度)
        self.assertAlmostEqual(dv.measure_angle((1, 0), (0, 0), (1, 2), 2, 1), 45)
        self.assertIsNone(dv.measure_angle((0, 0), (0, 0), (1, 1), 1, 1))

    def test_roi_stats(self):
        a = np.full((40, 40), 10, dtype=np.int16)
        a[15:25, 15:25] = 100
        st = dv.roi_stats(a, (10, 10), (30, 30), 0.5, 0.5)
        self.assertEqual(st["max"], 100)
        self.assertEqual(st["min"], 10)
        self.assertAlmostEqual(st["area"], st["n"] * 0.25)
        self.assertAlmostEqual(st["area"], math.pi * 10 * 10 * 0.25, delta=4)   # 半径10px の円
        flat = dv.roi_stats(np.full((40, 40), 7), (5, 5), (20, 20), 1, 1)
        self.assertEqual((flat["mean"], flat["std"]), (7, 0))
        self.assertIsNone(dv.roi_stats(a, (5, 5), (5.2, 5.2), 1, 1))


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.vol = np.arange(5 * 3 * 3, dtype=np.int16).reshape(5, 3, 3)

    def test_modes(self):
        v = self.vol
        self.assertTrue(np.array_equal(dv.slab_project(v, 0, 2, 3, "mip"), v[1:4].max(axis=0)))
        self.assertTrue(np.array_equal(dv.slab_project(v, 0, 2, 3, "minip"), v[1]))
        self.assertTrue(np.allclose(dv.slab_project(v, 0, 2, 3, "mean"), v[1:4].mean(axis=0)))
        self.assertTrue(np.array_equal(dv.slab_project(v, 0, 2, 3, "none"), v[2]))

    def test_edges_and_axes(self):
        v = self.vol
        self.assertTrue(np.array_equal(dv.slab_project(v, 0, 0, 5, "mip"), v[0:3].max(axis=0)))   # 端で切り詰め
        self.assertEqual(dv.slab_project(v, 2, 1, 3, "mip").shape, (5, 3))
        self.assertTrue(np.array_equal(dv.slab_project(v, 1, 1, 1, "mip"), v[:, 1, :]))


class LabelTest(unittest.TestCase):
    def test_orientation_letters(self):
        self.assertEqual(dv._letter(np.array([1, 0, 0])), "L")
        self.assertEqual(dv._letter(np.array([0, -1, 0])), "A")
        self.assertEqual(dv._letter(np.array([0, 0, -1])), "I")
        self.assertEqual(dv._opposite("L"), "R")

    def test_tissue(self):
        self.assertEqual(dv.tissue_label(-1000), "空気")
        self.assertEqual(dv.tissue_label(0), "水・髄液")
        self.assertEqual(dv.tissue_label(40), "軟部組織(脳・筋)")
        self.assertEqual(dv.tissue_label(1000), "骨")


if __name__ == "__main__":
    unittest.main()
