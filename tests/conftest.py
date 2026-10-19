import numpy as np
import pytest


@pytest.fixture
def make_label(tmp_path):
    def create(names, types, rows, *, filename="sample", constants=None, start="1999-03-01T00:00:00Z"):
        widths = [max(24, max(len(str(row[i])) for row in rows)) for i in range(len(names))]
        records = [" ".join(str(value).ljust(width) for value, width in zip(row, widths)) + "\r\n"
                   for row in rows]
        (tmp_path / f"{filename}.TAB").write_bytes("".join(records).encode("ascii"))
        fields = []
        location = 1
        for name, dtype, width in zip(names, types, widths):
            special = ""
            if constants and name in constants:
                special = "<Special_Constants>" + "".join(
                    f"<{key}>{value}</{key}>" for key, value in constants[name].items()) + "</Special_Constants>"
            fields.append(f"""<Field_Character><name>{name}</name>
                <field_location unit="byte">{location}</field_location><data_type>{dtype}</data_type>
                <field_length unit="byte">{width}</field_length><unit>nT</unit>{special}</Field_Character>""")
            location += width + 1
        path = tmp_path / f"{filename}.xml"
        path.write_text(f"""<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
            <Observation_Area><Time_Coordinates><start_date_time>{start}</start_date_time>
            </Time_Coordinates></Observation_Area><File_Area_Observational>
            <File><file_name>{filename}.TAB</file_name></File><Table_Character>
            <name>test_table</name><offset unit="byte">0</offset><records>{len(rows)}</records>
            <record_delimiter>Carriage-Return Line-Feed</record_delimiter><Record_Character>
            <fields>{len(names)}</fields><groups>0</groups>
            <record_length unit="byte">{len(records[0])}</record_length>
            {''.join(fields)}</Record_Character></Table_Character></File_Area_Observational>
            </Product_Observational>""")
        return path
    return create


@pytest.fixture
def synthetic_kernels(tmp_path):
    spice = pytest.importorskip("spiceypy")
    text = tmp_path / "geometry.tpc"
    text.write_text("""KPL/PCK
\\begindata
BODY301_RADII = ( 1737.4 1737.4 1737.4 )
BODY10_RADII = ( 696000 696000 696000 )
BODY301_POLE_RA = ( 0 0 0 )
BODY301_POLE_DEC = ( 90 0 0 )
BODY301_PM = ( 0 0 0 )
BODY301_LONG_AXIS = ( 0 )
DELTET/DELTA_T_A = 32.184
DELTET/K = 1.657D-3
DELTET/EB = 1.671D-2
DELTET/M = ( 6.239996 1.99096871D-7 )
DELTET/DELTA_AT = ( 32, @1999-JAN-1 )
\\begintext
""")
    spk = tmp_path / "sun.bsp"
    handle = spice.spkopn(str(spk), "synthetic geometry", 0)
    states = np.array([[1.5e8, 0, 0, 0, 0, 0], [1.5e8, 0, 0, 0, 0, 0]], dtype=float)
    spice.spkw08(handle, 10, 301, "J2000", -1e9, 1e9, "synthetic sun", 1, 2, states, -1e9, 2e9)
    spice.spkw08(handle, 301, 0, "J2000", -1e9, 1e9, "synthetic moon", 1, 2,
                 np.zeros((2, 6)), -1e9, 2e9)
    spice.spkcls(handle)
    return [text, spk]
