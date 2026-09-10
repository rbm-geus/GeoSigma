import struct
import numpy as np

def ReadSurfer7(filename):
    with open(filename, 'rb') as f:
        print("Reading GRD File")

        # --- File type ---
        d = f.read(4).decode('ascii')
        print(f"FileType - {d}")

        if d != "DSRB":
            raise ValueError("File format not supported!")
        else:
            print("File format supported!")

        # --- Header size and version ---
        HeaderSize = struct.unpack('<i', f.read(4))[0]
        HeaderVersion = struct.unpack('<i', f.read(4))[0]

        # --- Section name (expected: GRID) ---
        SectionName = f.read(4).decode('ascii')
        if SectionName == "GRID":
            print("Reading Grid Section")

        # --- Grid section length ---
        sec2 = struct.unpack('<i', f.read(4))[0]

        # --- Grid size (rows, cols) ---
        sec3 = struct.unpack('<2i', f.read(8))
        GridRows, GridColumns = sec3
        print(f"Grid Size: {GridRows} rows and {GridColumns} columns")

        # --- Lower-left corner coordinates ---
        sec4 = struct.unpack('<2d', f.read(16))
        GridxLL, GridyLL = sec4
        print(f"xLL: {GridxLL}, yLL: {GridyLL}")

        # --- Grid resolution ---
        sec5 = struct.unpack('<2d', f.read(16))
        GridResolutionX, GridResolutionY = sec5
        print(f"x-size: {GridResolutionX} m, y-size: {GridResolutionY} m")

        # --- Min/Max Z ---
        sec6 = struct.unpack('<2d', f.read(16))
        Min_Z, Max_Z = sec6
        print(f"Min. Z: {Min_Z} m, Max. Z: {Max_Z} m")

        # --- Grid rotation ---
        GridRotation = struct.unpack('<d', f.read(8))[0]

        # --- Blank value ---
        BlankValue = struct.unpack('<d', f.read(8))[0]

        # --- Section name (expected: DATA) ---
        SectionName2 = f.read(4).decode('ascii')

        # --- Data section length ---
        DataSectionLength = struct.unpack('<i', f.read(4))[0]
        print(f"Data Contains {DataSectionLength / 1024:.2f} kilobytes")

        # --- Data section ---
        print("Reading Data")
        num_points = DataSectionLength // 8  # each double = 8 bytes
        Data = np.fromfile(f, dtype='<d', count=num_points)

        # Replace blanks with NaN
        Data[Data == BlankValue] = np.nan

        # Reshape as per Surfer structure.
        # MATLAB's ReadSurfer7.m does ``reshape(Data,[Cols,Rows])'`` where MATLAB
        # reshape is COLUMN-major (Fortran order). NumPy reshape defaults to
        # row-major (C order), so the literal ``reshape((Cols,Rows)).T`` scrambles
        # the data for any non-square grid (it produced striped, mis-classed
        # output). The C-order reshape to (Rows, Cols) reproduces MATLAB exactly:
        # both give Z[r, c] = Data[r*Cols + c].
        Z = Data.reshape((GridRows, GridColumns))

        # --- Output variables ---
        DataOut = Z

        Info = {
            'UTM_X': GridxLL + np.arange(GridColumns) * GridResolutionX,
            'UTM_Y': GridyLL + np.arange(GridRows) * GridResolutionY,
            'BlankValue': BlankValue,
            'xLL': GridxLL,
            'yLL': GridyLL,
            'Cols': GridColumns,
            'Rows': GridRows,
            'MaxZ': Max_Z,
            'MinZ': Min_Z,
            'xResolution': GridResolutionX,
            'yResolution': GridResolutionY,
            'GridRotation': GridRotation,
        }

        print("File Loaded Successfully")

    return DataOut, Info
