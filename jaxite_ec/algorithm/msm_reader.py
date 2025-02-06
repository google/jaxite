"""Reads scalar and point files from the MSM algorithm."""

import re


class MSMReader:
  """Reads scalar and point files from the MSM algorithm."""

  def __init__(
      self, scalar_file_path: str, base_file_path: str, result_file_path: str
  ) -> None:
    self.scalar_file = open(scalar_file_path, 'r')
    self.base_file = open(base_file_path, 'r')
    self.result_file = open(result_file_path, 'r')

  def get_next_scalar(self):
    line = self.scalar_file.readline()
    if not line:
      return None
    else:
      return self.process_scalar_line(line)

  def get_next_base(self):
    line = self.base_file.readline()
    if not line:
      return None
    else:
      return self.process_point_line(line)

  def get_result(self):
    line = self.result_file.readline()
    return self.process_point_line(line)

  def process_scalar_line(self, line: str):
    match = re.search(r'\(([^)]+)\)', line)
    assert match
    scalar = int(match.group(1), 16)
    return scalar

  def process_point_line(self, line: str):
    matches = re.findall(r'\(([^)]+)\)', line)
    assert matches
    x = int(matches[0], 16)
    y = int(matches[1], 16)
    return [x, y]

  def close_files(self):
    self.scalar_file.close()
    self.base_file.close()
    self.result_file.close()
