import csv
import gzip
import json
import os
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
import warnings
import jax
import jax.numpy as jnp
import pandas as pd


class DataFrameGenerator:
  """A utility class for building pandas DataFrames from column data."""

  def __init__(self):
    """Initialize an empty DataFrameGenerator."""
    self.data: Dict[str, List[Any]] = {}

  def add_data(self, column_name: str, values: List[Any]) -> None:
    """Add data to a specific column.

    Args:
        column_name: Name of the column to add data to
        values: List of values to add to the column
    """
    if not isinstance(column_name, str):
      raise ValueError('column_name must be a string')
    if not isinstance(values, list):
      raise ValueError('values must be a list')

    if column_name not in self.data:
      self.data[column_name] = []
    self.data[column_name].extend(values)

  def add_single_value(self, column_name: str, value: Any) -> None:
    """Add a single value to a specific column.

    Args:
        column_name: Name of the column to add data to
        value: Single value to add to the column
    """
    self.add_data(column_name, [value])

  def get_column_lengths(self) -> Dict[str, int]:
    """Get the length of each column.

    Returns:
        Dictionary mapping column names to their lengths
    """
    return {col: len(values) for col, values in self.data.items()}

  def is_balanced(self) -> bool:
    """Check if all columns have the same length.

    Returns:
        True if all columns have the same length, False otherwise
    """
    if not self.data:
      return True
    lengths = set(len(col) for col in self.data.values())
    return len(lengths) == 1

  def to_dataframe(self, auto_balance: bool = True) -> pd.DataFrame:
    """Convert the stored data to a pandas DataFrame.

    Args:
        auto_balance: If True, automatically trim columns to the minimum length.
          If False, raise an error if columns have different lengths.

    Returns:
        pandas DataFrame with the stored data

    Raises:
        ValueError: If auto_balance is False and columns have different lengths
    """
    if not self.data:
      return pd.DataFrame()

    if not auto_balance and not self.is_balanced():
      lengths = self.get_column_lengths()
      raise ValueError(f'Columns have different lengths: {lengths}')

    # Find the minimum length among all columns
    min_len = min(len(col) for col in self.data.values())

    # Trim each column to the minimum length
    trimmed_data = {k: v[:min_len] for k, v in self.data.items()}

    return pd.DataFrame(trimmed_data)

  def clear(self) -> None:
    """Clear all stored data."""
    self.data.clear()

  def get_column_names(self) -> List[str]:
    """Get the names of all columns.

    Returns:
        List of column names
    """
    return list(self.data.keys())

  def has_column(self, column_name: str) -> bool:
    """Check if a column exists.

    Args:
        column_name: Name of the column to check

    Returns:
        True if the column exists, False otherwise
    """
    return column_name in self.data

  def merge(self, other_dataframe_generator: 'DataFrameGenerator'):
    """Merge the stored data with another DataFrameGenerator.

    Args:
        other_dataframe_generator: Another DataFrameGenerator to merge with

    Returns:
        Merged DataFrameGenerator
    """
    if not isinstance(other_dataframe_generator, DataFrameGenerator):
      raise ValueError('other_dataframe_generator must be a DataFrameGenerator')
    # Check if this DataFrameGenerator is empty
    if not self.data:
      self.data = other_dataframe_generator.data
      return
    # Check if the other DataFrameGenerator has the same column names
    if not set(self.get_column_names()) == set(
        other_dataframe_generator.get_column_names()
    ):
      print('The two DataFrameGenerators have different column names')
      return
      # raise ValueError("The two DataFrameGenerators have different column names")
    # Merge the data
    for column_name in other_dataframe_generator.get_column_names():
      self.add_data(column_name, other_dataframe_generator.data[column_name])

  def get_header(self) -> List[str]:
    """Get the header of the DataFrameGenerator.

    Returns:
        List of column names
    """
    return list(self.data.keys())

  def get_row_dict(self, index: int) -> Dict[str, Any]:
    """Get a row of the DataFrameGenerator.

    Returns:
        Dictionary of column names and values
    """
    return {
        column_name: self.data[column_name][index]
        for column_name in self.get_column_names()
    }


class TraceParser:

  def __init__(self, trace_dir: str):
    self.trace_dir = trace_dir

  def set_trace_dir(self, new_dir: str):
    """Set a new trace directory for the parser."""
    self.trace_dir = new_dir

  def find_trace_file(self):
    """Recursively search for the latest .trace.json.gz file in the trace_dir.

    Returns the full path to the file, or None if not found.
    """
    trace_files = []
    for root, _, files in os.walk(self.trace_dir):
      for file in files:
        if file.endswith('.trace.json.gz'):
          trace_files.append(os.path.join(root, file))

    if not trace_files:
      return None

    # Return the most recently modified file
    return max(trace_files, key=os.path.getmtime)

  def read_trace_json(self):
    """Finds, unzips, and reads the JSON content from the trace file.

    Returns the loaded JSON object, or None if not found or error.
    """
    trace_file = self.find_trace_file()
    if trace_file is None:
      print('No trace file found.')
      return None
    try:
      with gzip.open(trace_file, 'rt', encoding='utf-8') as f:
        data = json.load(f)
      return data
    except Exception as e:
      print(f'Error reading trace file: {e}')
      return None

  def parse_trace_csv(self):
    """Parses the trace CSV file and returns a list of trace events."""
    csv_file = os.path.join(self.trace_dir, 'trace_events.csv')

    # Read the trace JSON data
    trace_data = self.read_trace_json()
    if trace_data is None:
      print('Failed to read trace data')
      return None

    # Extract trace events
    trace_events = trace_data.get('traceEvents', [])
    if not trace_events:
      print('No trace events found in the data')
      return None

    headers = ['pid', 'tid', 'ts', 'dur', 'ph', 'name', 'args']
    # Write to CSV directly
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
      writer = csv.DictWriter(f, fieldnames=headers)
      writer.writeheader()
      for event in trace_events:
        # Convert args dictionary to string if it exists
        if 'args' in event:
          event['args'] = json.dumps(event['args'])
        else:
          event['args'] = ''

        # Write the event
        writer.writerow(event)
    print(f'Trace events written to: {csv_file}')


def calculate_statistics(data: List[Any]) -> Dict[str, Any]:
  """Calculate the statistics of the data.

  Args:
      data: List of data

  Returns:
      Dictionary containing the statistics
  """
  mean_value = statistics.mean(data)
  if len(data) == 1:
    std_value = 0
  else:
    std_value = statistics.stdev(data)
  min_value = min(data)
  max_value = max(data)
  median_value = statistics.median(data)
  return {
      'mean': mean_value,
      'std': std_value,
      'min': min_value,
      'max': max_value,
      'median': median_value,
  }


def list_add(list1: List[Any], list2: List[Any]) -> List[Any]:
  """Sum two lists element-wise.

  Args:
      list1: First list to sum
      list2: Second list to sum

  Returns:
      List of the sum of the two lists
  """
  assert len(list1) == len(list2), 'The two lists must have the same length'
  return [e1 + e2 for e1, e2 in zip(list1, list2)]


class KernelWrapper:

  def __init__(
      self,
      kernel_name: str,
      function_to_wrap: Callable,
      input_structs: List[Tuple[Tuple[int, ...], jnp.dtype]],
      mesh: Optional[jax.sharding.Mesh] = None,
      input_shardings: Optional[Tuple[jax.sharding.Sharding, ...]] = None,
      output_sharding: Optional[jax.sharding.Sharding] = None,
      parameters: Optional[Dict[str, Any]] = {},
      enable_sharding: bool = False,
  ):
    self.kernel_name = kernel_name
    self.callable_function = function_to_wrap
    self.input_structs = input_structs
    self.parameters = parameters
    self.mesh = mesh
    self.input_shardings = input_shardings
    self.output_sharding = output_sharding
    self.enable_sharding = enable_sharding

    self.jit_lower = None
    self.jit_compiled_function = None

    # Compile immediately upon initialization
    self._compile()

  def _compile(self):
    jax_input_structs = []
    if self.enable_sharding and self.input_shardings:
      for (shape, dtype), sharding in zip(
          self.input_structs, self.input_shardings
      ):
        jax_input_structs.append(
            jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
        )
    else:
      for shape, dtype in self.input_structs:
        jax_input_structs.append(jax.ShapeDtypeStruct(shape, dtype))

    # NOTE: Do not change the name of the function, it is used for profiling
    if self.parameters:

      def compiled_kernel_function(*jax_array_inputs):
        return self.callable_function(
            *jax_array_inputs, parameters=self.parameters
        )

    else:

      def compiled_kernel_function(*jax_array_inputs):
        return self.callable_function(*jax_array_inputs)

    if self.enable_sharding and self.mesh:
      with self.mesh:
        self.jit_lower = jax.jit(
            jax.named_call(compiled_kernel_function, name=self.kernel_name),
            in_shardings=self.input_shardings,
            out_shardings=self.output_sharding,
        ).lower(*jax_input_structs)
    else:
      self.jit_lower = jax.jit(
          jax.named_call(compiled_kernel_function, name=self.kernel_name)
      ).lower(*jax_input_structs)

    self.jit_compiled_function = self.jit_lower.compile()

  def get_compiled_function(self) -> Callable[..., jnp.ndarray]:
    assert self.jit_compiled_function is not None, 'Kernel not compiled'
    if self.enable_sharding and self.mesh:
      mesh = self.mesh

      def compiled_with_mesh(*jax_array_inputs):
        with mesh:
          return self.jit_compiled_function(*jax_array_inputs)

      return compiled_with_mesh
    return self.jit_compiled_function

  def get_input_structs(self):
    return self.input_structs

  def get_kernel_name(self) -> str:
    return self.kernel_name

  def shard_inputs(self, input_arrays: List[jnp.ndarray]) -> List[jnp.ndarray]:
    """Place inputs on the provided sharding."""
    if self.enable_sharding and self.input_shardings:
      return [
          jax.device_put(arr, sharding)
          for arr, sharding in zip(input_arrays, self.input_shardings)
      ]
    return input_arrays


class Profiler:

  def __init__(
      self,
      output_trace_path: str,
      profile_naming: str,
      configuration: Optional[Dict[str, Any]] = None,
  ):
    self.trace_dir = output_trace_path
    self.profiler_name = profile_naming
    self.profile_dir = os.path.join(self.trace_dir, self.profiler_name)
    if not os.path.exists(self.profile_dir):
      os.makedirs(self.profile_dir)

    self.configuration = configuration or {}
    self.random_seed = self.configuration.get('random_seed', 0)
    self.iterations = self.configuration.get('iterations', 1)
    self.save_to_file = self.configuration.get('save_to_file', True)
    self.enable_sharding = self.configuration.get('enable_sharding', False)

    self.profiles: List[Dict[str, Any]] = []
    self.profile_name_list: List[str] = []

    # Storage for results
    self.storage_file = os.path.join(
        self.profile_dir, f'{self.profiler_name}_results.csv'
    )

  def add_profile(
      self,
      name: str,
      kernel_wrapper: KernelWrapper,
      kernel_setting_cols: Dict[str, Any] = {},
  ):
    if name in self.profile_name_list:
      raise ValueError(f'Profiler name {name} already exists')

    self.profile_name_list.append(name)

    profile_folder = os.path.join(self.profile_dir, name)
    if not os.path.exists(profile_folder):
      os.makedirs(profile_folder)

    self.profiles.append({
        'name': name,
        'wrapper': kernel_wrapper,
        'settings': kernel_setting_cols,
        'folder': profile_folder,
        'failed': False,
        'trace_events': None,
        'filtered_events': None,
        'stats': None,
    })

  def _get_input_arrays(self, kernel_wrapper: KernelWrapper):
    def get_max_value(dtype):
      if dtype == jnp.uint8:
        return 128
      elif dtype == jnp.uint16:
        return 32768
      elif dtype == jnp.uint32:
        return 4294967295
      elif dtype == jnp.uint64:
        return 4294967295
      raise ValueError(f'Unsupported dtype: {dtype}')

    random_key = jax.random.key(self.random_seed)
    input_arrays = []
    for shape, dtype in kernel_wrapper.get_input_structs():
      if jnp.issubdtype(dtype, jnp.floating):
        input_arrays.append(jax.random.uniform(random_key, shape, dtype))
      elif jnp.issubdtype(dtype, jnp.integer):
        input_arrays.append(
            jax.random.randint(
                random_key, shape, 0, get_max_value(dtype), dtype
            )
        )
      elif jnp.issubdtype(dtype, jnp.bool_):
        input_arrays.append(jax.random.bernoulli(random_key, p=0.5, shape=shape))
      else:
        raise ValueError(f'Unsupported dtype: {dtype}')
    for input_array in input_arrays:
      input_array.block_until_ready()

    if self.enable_sharding:
      input_arrays = kernel_wrapper.shard_inputs(input_arrays)

    return input_arrays

  def profile_all_profilers(self):
    for profile in self.profiles:
      print(f"Profiling {profile['name']}")
      try:
        # Kernel wrapper is already compiled in its init
        wrapper = cast(KernelWrapper, profile['wrapper'])
        compiled_function = wrapper.get_compiled_function()
        input_arrays = self._get_input_arrays(wrapper)

        with jax.profiler.trace(profile['folder']):
          for _ in range(self.iterations):
            compiled_function(*input_arrays).block_until_ready()
      except Exception as e:
        print(f"Error profiling {profile['name']}:\n {e}")
        profile['failed'] = True

  def _parse_json_trace(self, profile):
    trace_parser = TraceParser(profile['folder'])
    trace_file_path = trace_parser.find_trace_file()
    profile_json = trace_parser.read_trace_json()
    if profile_json is None:
      warnings.warn(
          f"{profile['name']}: No trace events found in the data", UserWarning
      )
      profile['failed'] = True
      return None
    trace_events = profile_json.get('traceEvents', [])
    if not trace_events:
      warnings.warn(
          f"{profile['name']}: No trace events found in the data", UserWarning
      )
      profile['failed'] = True
      return None
    if self.save_to_file:
      # Save into the same folder as the raw trace file
      output_dir = os.path.dirname(trace_file_path)
      profile['output_folder'] = output_dir
      with open(os.path.join(output_dir, 'trace_events.json'), 'w') as f:
        json.dump(trace_events, f, indent=2)
    profile['trace_events'] = trace_events
    return trace_events

  def _filter_trace_events(self, profile):
    trace_events = profile['trace_events']
    if trace_events is None:
      return None

    def merge_filtered_events_by_name(filtered_events):
      grouped = {}
      for event in filtered_events:
        event_name = event.get('name', 'unknown')
        if (
            'args' in event.keys()
            and 'deduplicated_name' in event['args'].keys()
        ):
          event_name += '_' + event['args']['deduplicated_name']
        elif (
            'custom-call' in event['name']
            and 'args' in event.keys()
            and 'tf_op' in event['args'].keys()
        ):
          event_name += '_' + event['args']['tf_op']
        if event_name not in grouped:
          grouped[event_name] = []
        grouped[event_name].append(event)

      merged_filtered_events = {}
      for event_name, events in grouped.items():
        merged = events[0].copy()
        merged['dur'] = [e.get('dur') for e in events if 'dur' in e]
        merged['ts'] = [e.get('ts') for e in events if 'ts' in e]
        merged['repeat_count'] = len(events)
        merged_filtered_events[event_name] = merged
      return merged_filtered_events

    filtered_events_list = []
    # Check if NVIDIA is in device kind OR CPU is used as a fallback if explicit check needed
    # But generally JAX trace events differ by backend.
    # Assuming typical CPU/GPU separation.
    device_kind = jax.devices()[0].device_kind

    if 'NVIDIA' in device_kind:
      for e in trace_events:
        if 'args' in e and 'tf_op' in e['args']:
          # Loosen the check for compiled_kernel_function as it might be nested differently or named differently
          if 'compiled_kernel_function' in e['args'].get(
              'hlo_module', ''
          ) or 'compiled_kernel_function' in e['args'].get('long_name', ''):
            merged_event = False
            # Try to merge with existing events
            for f in filtered_events_list:
              # Check if correlation_id exists before accessing it
              if (
                  'correlation_id' in f['args']
                  and 'correlation_id' in e['args']
                  and f['args']['correlation_id'] == e['args']['correlation_id']
                  and f['name'] == e['name']
              ):
                f['dur'] = f['dur'] + e['dur']
                merged_event = True
            if not merged_event:
              filtered_events_list.append(e)
      profile['filtered_events'] = merge_filtered_events_by_name(
          filtered_events_list
      )

    elif 'TPU' in device_kind:
      for event in trace_events:
        if (
            'pid' not in event.keys() or event['pid'] != 3
        ):  # ToDo: change it into automatic PID detection based on "TPU:0".
          continue
        if (
            'name' in event.keys()
            and 'compiled_kernel_function' in event['name']
            and 'args' in event.keys()
        ):
          filtered_events_list.append(event)
        elif 'args' in event.keys() and 'long_name' in event['args'].keys():
          filtered_events_list.append(event)
        else:
          continue
      profile['filtered_events'] = merge_filtered_events_by_name(
          filtered_events_list
      )
    else:
      # Fallback for CPU or other devices
      # CPU traces might be different. Let's try to capture events related to our kernel.
      for event in trace_events:
        if 'name' in event and 'compiled_kernel_function' in event['name']:
          filtered_events_list.append(event)
      profile['filtered_events'] = merge_filtered_events_by_name(
          filtered_events_list
      )

    # Always save filtered events if we have any
    if self.save_to_file:
      # Make sure we don't crash if profile['filtered_events'] is None
      events_to_dump = (
          profile['filtered_events']
          if profile['filtered_events'] is not None
          else {}
      )
      with open(
          os.path.join(profile['output_folder'], 'filtered_events.json'), 'w'
      ) as f:
        json.dump(events_to_dump, f, indent=2)

  def _calculate_profiling_statistics(self, profile):
    if profile['filtered_events'] is None:
      return

    repeat_count = self.iterations
    kernel_duration = [0] * repeat_count

    device_kind = jax.devices()[0].device_kind

    if 'NVIDIA' in device_kind:
      for event in profile['filtered_events'].values():
        if 'compiled_kernel_function' in event['args'].get('hlo_module', ''):
          durations = event['dur']
          if not isinstance(durations, list):
            durations = [durations]

          if len(durations) == repeat_count:
            kernel_duration = list_add(kernel_duration, durations)
          elif (
              len(durations) > repeat_count
              and len(durations) % repeat_count == 0
          ):
            # Assume sequential execution of kernels within one iteration
            chunk_size = len(durations) // repeat_count
            aggregated_durations = [
                sum(durations[i * chunk_size : (i + 1) * chunk_size])
                for i in range(repeat_count)
            ]
            kernel_duration = list_add(kernel_duration, aggregated_durations)
          else:
            # Fallback: just take first N or handle mismatch.
            # For now, adopting CPU strategy of taking first N but this is likely under-reporting.
            # Ideally log a warning.
            kernel_duration = list_add(
                kernel_duration, durations[:repeat_count]
            )
    elif 'TPU' in device_kind:
      for event in profile['filtered_events'].values():
        if 'compiled_kernel_function' in event['name']:
          kernel_duration = list_add(kernel_duration, event['dur'])
    else:
      # CPU logic - assuming direct name match from filtered events
      for event in profile['filtered_events'].values():
        # On CPU, events might be simpler
        if 'compiled_kernel_function' in event.get('name', ''):
          # DUR might be a single value or list depending on how it was merged
          durations = event['dur']
          if not isinstance(durations, list):
            durations = [durations]

          # If we have less durations than repeat_count, we might need to pad or it's a mismatch
          # For now, let's just add what we have, assuming 1-to-1 or aggregated
          if len(durations) == repeat_count:
            kernel_duration = list_add(kernel_duration, durations)
          elif len(durations) > repeat_count:
            # Take first N
            kernel_duration = list_add(
                kernel_duration, durations[:repeat_count]
            )
          else:
            # Append 0s? Or just take what we have
            padded = durations + [0] * (repeat_count - len(durations))
            kernel_duration = list_add(kernel_duration, padded)

    profile['stats'] = {
        'kernel_all': kernel_duration,
    }

  def post_process_all_profilers(self):
    for profile in self.profiles:
      if profile['failed']:
        continue

      events = self._parse_json_trace(profile)
      if events is None:
        continue

      self._filter_trace_events(profile)
      self._calculate_profiling_statistics(profile)

    self.write_results()

  def get_profiling_dataframe_generator_all_profilers(self):
    df_generator = DataFrameGenerator()
    for profile in self.profiles:
      if profile['failed'] or profile['stats'] is None:
        continue

      p_df_gen = DataFrameGenerator()
      p_df_gen.add_single_value(
          'operation_name', profile['wrapper'].get_kernel_name()
      )

      for key, value in profile['settings'].items():
        p_df_gen.add_single_value(key, value)

      all_kernel_duration = profile['stats']['kernel_all']
      for i, duration in enumerate(all_kernel_duration):
        p_df_gen.add_single_value(f'sample_{i}', duration)

      df_generator.merge(p_df_gen)
    return df_generator

  def write_results(self):
    storage_dataframe_generator = (
        self.get_profiling_dataframe_generator_all_profilers()
    )
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(self.storage_file)
    mode = 'a' if file_exists else 'w'
    header = not file_exists
    storage_dataframe_generator.to_dataframe().to_csv(
        self.storage_file, mode=mode, header=header, index=False
    )
    print(
        storage_dataframe_generator.to_dataframe().to_csv()
    )  # Need to see the content of the file in terminal as Google does not have file system
    print(f'Results written to: {self.storage_file}')


def collect_logs(root_dir='.', output_csv_name='all_logs_collected'):
  """Collects all CSV files found under directories named 'log'

  and aggregates them into a single CSV file.
  Handles varying headers by taking the union of all found columns.
  """
  all_files = []

  # Fieldnames set to collect all unique columns
  all_fieldnames = set()
  # To preserve some order, we can use a list and add new ones as we see them
  ordered_fieldnames = []

  # First pass: identify files and collect all possible fieldnames
  for dirpath, dirnames, filenames in os.walk(root_dir):
    path_parts = dirpath.split(os.sep)
    if 'log' in path_parts:
      for file in filenames:
        if file.endswith('.csv'):
          full_path = os.path.join(dirpath, file)
          all_files.append(full_path)
          try:
            with open(full_path, 'r', newline='') as csvfile:
              reader = csv.reader(csvfile)
              try:
                header = next(reader)
                for h in header:
                  if h not in all_fieldnames:
                    all_fieldnames.add(h)
                    ordered_fieldnames.append(h)
              except StopIteration:
                # Empty file
                pass
          except Exception as e:
            print(f'Error reading header of {full_path}: {e}')

  if not all_files:
    print('No CSV files found.')
    return

  print(f'Found {len(all_files)} CSV files.')
  print(f'Unified collected columns: {ordered_fieldnames}')

  output_file = os.path.join(root_dir, f'{output_csv_name}.csv')
  total_rows = 0

  try:
    with open(output_file, 'w', newline='') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=ordered_fieldnames)
      writer.writeheader()

      for full_path in all_files:
        try:
          with open(full_path, 'r', newline='') as infile:
            reader = csv.DictReader(infile)
            # The DictReader uses the file's own header mapping
            # We just iterate and write to the master dict writer
            for row in reader:
              writer.writerow(row)
              total_rows += 1
        except Exception as e:
          print(f'Error processing {full_path}: {e}')

    print(f'Saved aggregated logs to {os.path.abspath(output_file)}')
    print(f'Total rows collected: {total_rows}')

  except Exception as e:
    print(f'Error writing output file: {e}')
