# NOTE: We could have used NeuroConv to convert the ephys data but we want to use the Frank Lab's
# ndx-franklab-novela extension to store the Probe information for maximal integration with Spyglass,
# so we are doing the conversion manually using PyNWB.

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from hdmf.backends.hdf5 import H5DataIO
import numpy as np
from neuroconv.tools.spikeinterface.spikeinterfacerecordingdatachunkiterator import (
    SpikeInterfaceRecordingDataChunkIterator,
)
from pynwb import NWBFile
from pynwb.ecephys import ElectricalSeries
from spikeinterface.extractors import OpenEphysBinaryRecordingExtractor

MICROVOLTS_PER_VOLT = 1e6
VOLTS_PER_MICROVOLT = 1 / MICROVOLTS_PER_VOLT

MIN_IMPEDANCE_OHMS = 1e5
MAX_IMPEDANCE_OHMS = 3e6


def add_electrode_data(
    *,
    nwbfile: NWBFile,
    filtering_list: list[str],
    headstage_channel_numbers: list[int],
    reference_daq_channel_indices: list[int],
    metadata: dict,
):
    """
    Add the electrode data from the impedance and channel geometry files.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file being assembled.
    filtering_list : list of str
        The filtering applied to each channel.
    headstage_channel_numbers : list of int
        The headstage channel numbers for each channel.
    reference_daq_channel_indices : list of int
        The reference DAQ channel numbers for each channel.
    metadata : dict
        Metadata dictionary.
    """

    device_args = metadata["device"]
    assert set(device_args.keys()) == {
        "name",
        "description",
        "manufacturer",
    }, "Device arguments do not match expected keys."
    device = nwbfile.create_device(**device_args)

    electrodes_location = metadata["electrodes_location"]

    # Create an NWB ElectrodeGroup for all electrodes
    # TODO: confirm that all electrodes are in the same group
    electrode_group = nwbfile.create_electrode_group(
        name="ElectrodeGroup",
        description="All electrodes",
        location=electrodes_location,
        device=device,
    )

    impedance_file_path = metadata["impedance_file_path"]
    channel_geometry_file_path = metadata["channel_geometry_file_path"]
    ecog_channel_ids = metadata.get("ecog_channel_ids", [])

    electrode_data = pd.read_csv(impedance_file_path)

    # Check that the expected columns are present in order and no extra columns are present
    expected_columns = [
        "Channel Number",
        "Channel Name",
        "Port",
        "Enabled",
        "Impedance Magnitude at 1000 Hz (ohms)",
        "Impedance Phase at 1000 Hz (degrees)",
        "Series RC equivalent R (Ohms)",
        "Series RC equivalent C (Farads)",
    ]
    assert electrode_data.columns.tolist() == expected_columns, "Impedance file does not have the expected columns."

    # Check that the filtering list has the same length as the number of channels
    assert len(filtering_list) == len(
        electrode_data
    ), "Filtering list does not have the same length as the number of channels."

    # Drop the first column which should be the same as the second column
    assert (
        electrode_data["Channel Number"] == electrode_data["Channel Name"]
    ).all(), "First column is not the same as the second column."
    electrode_data.drop(columns=["Channel Number"], inplace=True)

    channel_geometry = pd.read_csv(channel_geometry_file_path, header=None)
    # The first column is the relative x coordinate
    # The second column is the relative y coordinate
    # Append the x and y coordinates to the impedance data
    assert len(channel_geometry) == len(
        electrode_data
    ), "Channel geometry and impedance data do not have the same length."
    electrode_data["rel_x"] = channel_geometry.iloc[:, 0]
    electrode_data["rel_y"] = channel_geometry.iloc[:, 1]

    # Mark electrodes with impedance that is less than 0.1 MOhms or more than 3.0 MOhms
    # as bad electrodes
    electrode_data["bad_channel"] = (electrode_data["Impedance Magnitude at 1000 Hz (ohms)"] < MIN_IMPEDANCE_OHMS) | (
        electrode_data["Impedance Magnitude at 1000 Hz (ohms)"] > MAX_IMPEDANCE_OHMS
    )

    # Mark ECoG electrodes -- ecog_channel_ids is 1-indexed
    electrode_data["ecog"] = np.isin(np.arange(1, len(electrode_data) + 1), ecog_channel_ids)

    # Add the electrode data to the NWB file, one column at a time
    nwbfile.add_electrode_column(
        name="channel_name",
        description="The name of the channel from the impedance file",
    )
    nwbfile.add_electrode_column(
        name="port",
        description="The port of the electrode from the impedance file",
    )
    nwbfile.add_electrode_column(
        name="enabled",
        description="Whether the channel is enabled from the impedance file",
    )
    nwbfile.add_electrode_column(
        name="imp",
        description="The impedance of the electrode (Impedance Magnitude at 1000 Hz (ohms))",
    )
    nwbfile.add_electrode_column(
        name="imp_phase",
        description="The phase of the impedance of the electrode (Impedance Phase at 1000 Hz (degrees))",
    )
    nwbfile.add_electrode_column(
        name="series_resistance_in_ohms",
        description="The series resistance of the electrode (Series RC equivalent R (Ohms))",
    )
    nwbfile.add_electrode_column(
        name="series_capacitance_in_farads",
        description="The series capacitance of the electrode (Series RC equivalent C (Farads))",
    )
    nwbfile.add_electrode_column(
        name="bad_channel",
        description="Whether the channel is a bad channel based on too low or too high impedance",
    )
    nwbfile.add_electrode_column(
        name="ecog",
        description="Whether the channel is an ECoG electrode",
    )
    nwbfile.add_electrode_column(
        name="rel_x",
        description="The relative x coordinate of the electrode",
    )
    nwbfile.add_electrode_column(
        name="rel_y",
        description="The relative y coordinate of the electrode",
    )
    nwbfile.add_electrode_column(
        name="filtering",
        description="The filtering applied to the electrode",
    )
    nwbfile.add_electrode_column(
        name="headstage_channel_number",
        description="The headstage channel number for the electrode",
    )
    nwbfile.add_electrode_column(
        name="reference_daq_channel_index",
        description="The index of the reference channel in this table. -1 if not set.",
    )

    assert len(filtering_list) == len(
        electrode_data
    ), "Filtering list does not have the same length as the number of channels."
    assert len(headstage_channel_numbers) == len(
        electrode_data
    ), "Headstage channel numbers do not have the same length as the number of channels."
    assert len(reference_daq_channel_indices) == len(
        electrode_data
    ), "Reference DAQ channel indices do not have the same length as the number of channels."

    for i, row in electrode_data.iterrows():
        nwbfile.add_electrode(
            channel_name=row["Channel Name"],
            port=row["Port"],
            enabled=bool(row["Enabled"]),
            imp=row["Impedance Magnitude at 1000 Hz (ohms)"],
            imp_phase=row["Impedance Phase at 1000 Hz (degrees)"],
            series_resistance_in_ohms=row["Series RC equivalent R (Ohms)"],
            series_capacitance_in_farads=row["Series RC equivalent C (Farads)"],
            bad_channel=row["bad_channel"],
            ecog=row["ecog"],
            rel_x=float(row["rel_x"]),
            rel_y=float(row["rel_y"]),
            group=electrode_group,
            filtering=filtering_list[i],
            location=electrodes_location,
            headstage_channel_number=headstage_channel_numbers[i],
            reference_daq_channel_index=reference_daq_channel_indices[i],
        )


def get_raw_ephys_data(
    folder_path: Path,
) -> tuple[SpikeInterfaceRecordingDataChunkIterator, float, np.ndarray]:
    """
    Get the raw ephys data from the OpenEphys binary recording.

    Parameters
    ----------
    folder_path : Path
        Path to the folder containing the OpenEphys binary recording. The folder
        should have the date in the name and contain a file called "settings.xml".

    Returns
    -------
    traces_as_iterator : SpikeInterfaceRecordingDataChunkIterator
        To be used as the data argument in pynwb.ecephys.ElectricalSeries.
    channel_conversion_factor : float
        The conversion factor from the raw data to volts.
    original_timestamps: np.ndarray
        The original timestamps from the recording. This could be used as the timestamps argument in
        pynwb.ecephys.ElectricalSeries or may need to be time aligned with the other
        data streams in the NWB file.
    """
    # Create a SpikeInterface recording extractor for the OpenEphys binary data
    # NOTE: We could write our own extractor to handle the relatively simple OpenEphys binary format
    # and surrounding files but it is nice to build on the well-tested code of others when possible.
    # However, we should remember that external code may not be maintained or have bugs of their own.
    recording = OpenEphysBinaryRecordingExtractor(folder_path=folder_path)

    # Select only the channels that start with "CH"
    # Ignore the "ADC" channels
    channel_ids_to_convert = [ch for ch in recording.channel_ids if ch.startswith("CH")]
    recording_sliced = recording.select_channels(channel_ids=channel_ids_to_convert)

    # Get the channel conversion factor
    channel_conversion_factors_uv = recording_sliced.get_channel_gains()
    # Warn if the channel conversion factors are not the same for all channels
    if not all(channel_conversion_factors_uv == channel_conversion_factors_uv[0]):
        warnings.warn(
            "The channel conversion factors are not the same for all channels. "
            "This is unexpected and may indicate a problem with the conversion factors."
        )
    channel_conversion_factor_v = channel_conversion_factors_uv[0] * VOLTS_PER_MICROVOLT

    # NOTE channel offsets should be 0 for all channels in openephys data
    channel_conversion_offsets = recording_sliced.get_channel_offsets()
    assert all(channel_conversion_offsets == 0), "Channel conversion offsets are not all 0."

    # Get the original timestamps
    original_timestamps = recording_sliced.get_times()

    # Create a SpikeInterfaceRecordingDataChunkIterator using all default buffering and
    # chunking options. This will be passed to the pynwb.ecephys.ElectricalSeries
    # constructor.
    traces_as_iterator = SpikeInterfaceRecordingDataChunkIterator(recording=recording_sliced)

    return (
        traces_as_iterator,
        channel_conversion_factor_v,
        original_timestamps,
    )


def get_raw_ephys_metadata(folder_path: Path) -> tuple[list[str], list[int], list[int]]:
    """
    Get the raw ephys metadata from the OpenEphys binary recording.

    Read the openephys settings.xml file to get the mapping of channel number to channel name.

    <PROCESSOR name="Sources/Rhythm FPGA" ...>
      <CHANNEL_INFO>
        <CHANNEL name="CH1" number="0" gain="0.19499999284744263"/>
        ...
      </CHANNEL_INFO>
      ...
    </PROCESSOR>

    Read the settings.xml file to get the filtering applied to each channel.

    Read the settings.xml file to get the channel map.

    Parameters
    ----------
    folder_path : Path
        Path to the folder containing the OpenEphys binary recording. The folder
        should have the date in the name and contain a file called "settings.xml".

    Returns
    -------
    filtering_list : list[str]
        The filtering applied to each channel.
    headstage_channel_numbers : list[int]
        The headstage channel numbers for each channel.
    reference_daq_channel_indices: list[int]
        The reference DAQ channel indices for each channel (-1 if not set).
    """
    settings_file_path = Path(folder_path) / "settings.xml"
    settings_tree = ET.parse(settings_file_path)
    settings_root = settings_tree.getroot()
    rhfp_processor = settings_root.find(".//PROCESSOR[@name='Sources/Rhythm FPGA']")
    if rhfp_processor is None:
        raise ValueError("Could not find the Rhythm FPGA processor in the settings.xml file.")
    channel_info = rhfp_processor.find("CHANNEL_INFO")
    if channel_info is None:
        raise ValueError("Could not find the CHANNEL_INFO node in the settings.xml file.")
    channel_number_to_channel_name = {
        channel.attrib["number"]: channel.attrib["name"] for channel in channel_info.findall("CHANNEL")
    }

    # Get the filtering info
    filtering_list = get_filtering_info(settings_root, channel_number_to_channel_name)

    # Get the channel map info
    headstage_channel_numbers, reference_daq_channel_indices = get_channel_map_info(
        settings_root, channel_number_to_channel_name
    )

    # TODO: save settings.xml as an associated file using the ndx-franklab-novela extension

    return (
        filtering_list,
        headstage_channel_numbers,
        reference_daq_channel_indices,
    )


def get_filtering_info(settings_root: ET.Element, channel_number_to_channel_name: dict[str, str]) -> list[str]:
    """
    Get the filtering applied to each channel from the settings.xml file.

    Read the settings.xml file to get the filtering applied to each channel - map channel number to filter description

    <PROCESSOR name="Filters/Bandpass Filter" ...>
      <CHANNEL name="0" number="0">
        <SELECTIONSTATE param="1" record="0" audio="0"/>
        <PARAMETERS highcut="6000" lowcut="1" shouldFilter="1"/>
      </CHANNEL>
      ...
    </PROCESSOR>

    Parameters
    ----------
    settings_root : ET.Element
        The root of the settings.xml file.
    channel_number_to_channel_name : dict[str, str]
        Mapping of channel number to channel name.

    Returns
    -------
    filtering_list: list[str]
        The filtering applied to each channel.
    """
    bandpass_filter = settings_root.find(".//PROCESSOR[@name='Filters/Bandpass Filter']")
    filtering = {}
    if bandpass_filter is not None:
        for channel in bandpass_filter.findall("CHANNEL"):
            # Ignore the ADC channels
            if not channel_number_to_channel_name.get(channel.attrib["number"]).startswith("CH"):
                continue
            highcut = channel.find("PARAMETERS").attrib["highcut"]
            lowcut = channel.find("PARAMETERS").attrib["lowcut"]
            should_filter = channel.find("PARAMETERS").attrib["shouldFilter"]
            if should_filter == "1":
                filtering[channel.attrib["number"]] = (
                    f"2nd-order Butterworth filter with highcut={highcut} Hz and lowcut={lowcut} Hz"
                )
            else:
                filtering[channel.attrib["number"]] = "No filtering"
                raise ValueError(f"Channel {channel.attrib['number']}: No filtering")
    else:
        raise ValueError("No bandpass filter found in the settings.xml file.")

    # Check that the channel numbers in filtering go from "0" to "N-1"
    # where N is the number of channels
    assert all(
        int(ch_num) == i for i, ch_num in enumerate(filtering.keys())
    ), "Channel numbers in filtering do not go from 0 to N-1."
    filtering_list = list(filtering.values())

    # Warn if the channel filtering is not the same for all channels
    if not all(f == filtering_list[0] for f in filtering_list):
        warnings.warn(
            "WARNING: The channel filtering is not the same for all channels. "
            "This is unexpected and may indicate a problem with the filtering settings."
        )

    return filtering_list


def get_channel_map_info(
    settings_root: ET.Element, channel_number_to_channel_name: dict[str, str]
) -> tuple[list[int], list[int]]:
    """
    Get the channel map info from the settings.xml file.

    Read the settings.xml file to get the mapping of daq / data channel index (0-indexed) to headstage channel
    index (1-indexed) and get the reference channels for each channel
    The GUI for this older version of the OpenEphys Channel Map filter is documented here:
    https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/950421/Channel+Map

    <PROCESSOR name="Filters/Channel Map" ...>
      <CHANNEL name="0" number="0">
        <SELECTIONSTATE param="0" record="0" audio="0"/>
      </CHANNEL>
      ...
      <EDITOR isCollapsed="0" displayName="Channel Map" Type="ChannelMappingEditor">
        <SETTING Type="visibleChannels" Value="6"/>
        <CHANNEL Number="0" Mapping="1" Reference="-1" Enabled="1"/>
        <CHANNEL Number="1" Mapping="2" Reference="-1" Enabled="1"/>
        <CHANNEL Number="2" Mapping="3" Reference="-1" Enabled="1"/>
        ...
        <REFERENCE Number="0" Channel="2"/>
        <REFERENCE Number="1" Channel="-1"/>
        ...
      </EDITOR>
    </PROCESSOR>

    Parameters
    ----------
    settings_root : ET.Element
        The root of the settings.xml file.
    channel_number_to_channel_name : dict[str, str]
        Mapping of channel number to channel name.

    Returns
    -------
    headstage_channel_numbers : list[int]
        The headstage channel numbers for each channel.
    reference_daq_channel_indices : list[int]
        The reference DAQ channel indices for each channel (-1 if not set).
    """
    channel_map = settings_root.find(".//PROCESSOR[@name='Filters/Channel Map']")
    if channel_map is None:
        raise ValueError("Could not find the Channel Map processor in the settings.xml file.")
    channel_map_editor = channel_map.find("EDITOR")
    if channel_map_editor is None:
        raise ValueError("Could not find the EDITOR node in the settings.xml file.")

    # Get the reference channel if set
    reference_channels = list()
    for i, reference in enumerate(channel_map_editor.findall("REFERENCE")):
        assert int(reference.attrib["Number"]) == i, "Reference number does not match index."
        reference_channels.append(int(reference.attrib["Channel"]))
        # Ryan believes the reference channel is the daq / data channel index and not the headstage channel index

    # Get the channel map
    headstage_channel_numbers = list()
    reference_daq_channel_indices = list()
    for i, channel in enumerate(channel_map_editor.findall("CHANNEL")):
        # There should not be any disabled channels and this code was not tested with disabled channels.
        # TODO Before removing this assertion, check that the code below works when a channel is disabled.
        assert int(channel.attrib["Enabled"]) == 1, "Channel is not enabled."
        assert int(channel.attrib["Number"]) == i, "Channel number does not match index."

        # Ignore the ADC channels
        if not channel_number_to_channel_name.get(channel.attrib["Number"]).startswith("CH"):
            continue

        # This code was not tested with reference channels.
        # TODO Before removing this assertion, check that the code below works when a channel is a reference.
        assert int(channel.attrib["Reference"]) == -1, "Channel is a reference."
        if int(channel.attrib["Reference"]) != -1:
            reference_daq_channel_indices.append(reference_channels[int(channel.attrib["Reference"])])
        else:
            reference_daq_channel_indices.append(-1)

        headstage_channel_numbers.append(int(channel.attrib["Mapping"]))

    return headstage_channel_numbers, reference_daq_channel_indices


def add_raw_ephys(
    *,
    nwbfile: NWBFile,
    metadata: dict = None,
) -> None:
    """Add the raw ephys data to a NWB file. Must be called after add_electrode_groups

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file being assembled.
    metadata : dict, optional
        Metadata dictionary.
    """
    print("Adding raw ephys...")
    openephys_folder_path = metadata["openephys_folder_path"]
    (
        traces_as_iterator,
        channel_conversion_factor_v,
        original_timestamps,
    ) = get_raw_ephys_data(openephys_folder_path)
    num_samples, num_channels = traces_as_iterator.maxshape

    (
        filtering_list,
        headstage_channel_numbers,
        reference_daq_channel_indices,
    ) = get_raw_ephys_metadata(openephys_folder_path)

    # Create electrode groups and add electrode data to the NWB file
    add_electrode_data(
        nwbfile=nwbfile,
        filtering_list=filtering_list,
        headstage_channel_numbers=headstage_channel_numbers,
        reference_daq_channel_indices=reference_daq_channel_indices,
        metadata=metadata,
    )

    # Check that the number of electrodes in the NWB file is the same as the number of channels in traces_as_iterator
    assert (
        len(nwbfile.electrodes) == num_channels
    ), "Number of electrodes in NWB file does not match number of channels in traces_as_iterator"

    # Create the electrode table region encompassing all electrodes
    electrode_table_region = nwbfile.create_electrode_table_region(
        region=list(range(len(nwbfile.electrodes))),
        description="Electrodes used in raw ElectricalSeries recording",
    )

    # A chunk of shape (81920, 64) and dtype int16 (2 bytes) is ~10 MB, which is the recommended chunk size
    # by the NWB team.
    # We could also add compression here. zstd/blosc-zstd are recommended by the NWB team, but
    # they require the hdf5plugin library to be installed. gzip is available by default.
    # Use gzip for now, but consider zstd/blosc-zstd in the future.
    data_data_io = H5DataIO(
        traces_as_iterator,
        chunks=(min(num_samples, 81920), min(num_channels, 64)),
        compression="gzip",
    )

    # Create the ElectricalSeries
    # For now, we do not chunk or compress the timestamps, which are relatively small
    # TODO: replace the timestamps with timestamps aligned with photometry data if photometry
    # data was also collected during the same recording session
    eseries = ElectricalSeries(
        name="ElectricalSeries",
        description=(
            "Raw ephys data from OpenEphys recording (multiply by conversion factor to get data in volts). "
            "Timestamps are the original timestamps from the OpenEphys recording."
        ),
        data=data_data_io,
        timestamps=original_timestamps,
        electrodes=electrode_table_region,
        conversion=channel_conversion_factor_v,
    )

    # Add the ElectricalSeries to the NWBFile
    nwbfile.add_acquisition(eseries)
