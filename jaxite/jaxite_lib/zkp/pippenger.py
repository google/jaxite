"""pippenger Implementation for Multi-scalar-multiplication (MSM) -- ZKP"""

from collections.abc import Sequence

from absl import app

from typing import List
from elliptic_curve import *
from config_file import USE_BARRETT, USE_GMP
if USE_GMP:
    from big_integer import GMPBigInteger as BigInt
else:
    from big_integer import BigInteger as BigInt
from msm_reader import MSMReader
import logging

class PippengerBucket:
    def __init__(self, coordinate_system: EllipticCurveCoordinateSystem, slice_id = 0):
        self.slice_id = slice_id
        #self.empty = True
        self.point = coordinate_system.generate_point(None, True)

    def add_point(self, point: ECCPoint):
            self.point += point

    def get_point(self):
        return self.point

class PippengerWindow:
    def __init__(self, coordinate_system: EllipticCurveCoordinateSystem, slice_length, window_id = 0) -> None:
        self.coordinate_system = coordinate_system
        self.slice_length = slice_length
        self.bucket_num = 2**slice_length
        self.window_id = window_id
        self.buckets:List[PippengerBucket] = [PippengerBucket(coordinate_system, i) for i in range(self.bucket_num)]
        self.point= coordinate_system.generate_point(None, True)

    def bucket_reduction(self) -> ECCPoint:
        window_sum: ECCPoint =  self.coordinate_system.generate_point(None, True)
        temp_sum: ECCPoint =  self.coordinate_system.generate_point(None, True)
        for i in range(self.bucket_num - 1, 0, -1):
            temp_sum += self.buckets[i].get_point()
            window_sum += temp_sum
        self.point = window_sum
        return window_sum
    
    def get_point(self):
        return self.point
    
    def __getitem__(self, index):
        return self.buckets[index]
    
    def __setitem__(self, index, value):
        self.buckets[index] = value




class PippengerMSM:
    def __init__(self,coordinate_system: EllipticCurveCoordinateSystem, slice_length, window_num) -> None:
        self.coordinate_system: EllipticCurveCoordinateSystem = coordinate_system
        self.slice_length = slice_length
        self.window_num = window_num
        self.bucket_num_in_window = 2**self.slice_length
        self.slice_mask =  self.bucket_num_in_window - 1
        self.bucket_num_all = self.bucket_num_in_window * self.window_num
        self.windows: List[PippengerWindow] = [PippengerWindow(coordinate_system, self.slice_length, i) for i in range(self.window_num)]
        self.result = None

    
    def msm_run(self, msm_reader: MSMReader):
        scalar = msm_reader.get_next_scalar()
        coordinates = msm_reader.get_next_base()

        while (scalar != None and coordinates != None):
            logging.debug(f"scalar: {hex(scalar)}")
            point = self.coordinate_system.generate_point(coordinates)
            self.msm_horizental_stream(scalar, point)
            scalar = msm_reader.get_next_scalar()
            coordinates = msm_reader.get_next_base()
        self.bucket_reduction()
        self.result = self.window_merge()
        self.result = self.result.coordinate_system.convert_to_affine()
        return self.result


    def msm_horizental_stream(self, scalar: int, point: ECCPoint):
        window_id = 0
        current_scalar = scalar
        while current_scalar != 0:
            logging.debug(f"window_id: {window_id}")
            bucket_id = current_scalar & self.slice_mask
            self.bucket_accumulation(window_id, bucket_id, point)
            logging.debug(f"bucket: {hex(bucket_id)}, {self.windows[window_id][bucket_id].get_point()}")
            current_scalar =  current_scalar >> self.slice_length
            window_id+=1

    def bucket_accumulation(self, window_id, bucket_id, point:ECCPoint):
        self.windows[window_id][bucket_id].add_point(point)

    def bucket_reduction(self):
        for i in range(0, self.window_num):
            logging.debug(f"window {i} BR")
            self.windows[i].bucket_reduction()
            logging.debug(f"{self.windows[i].get_point()}")

    def window_merge(self):
        merged = self.coordinate_system.generate_point(None, True)
        for i in range(self.window_num - 1, -1, -1):
            logging.debug(f"window {i} WM")
            point = self.windows[i].get_point()
            merged = (merged << self.slice_length) + point
        
        return merged
      
def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")


if __name__ == "__main__":
  app.run(main)
