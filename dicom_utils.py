import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, generate_uid
import datetime
import numpy as np

def create_dicom_dataset(pixel_data, filename, patient_name="Anonymous", patient_id="12345", 
                         modality="MR", series_description="Processed", 
                         slice_thickness=1.0, pixel_spacing=[1.0, 1.0], 
                         image_position=[0.0, 0.0, 0.0], image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                         instance_number=1, study_instance_uid=None, series_instance_uid=None):
    """
    Creates a pydicom Dataset for a single slice.
    """
    
    # Ensure pixel_data is correct type
    if pixel_data.dtype != np.uint16:
        # User might pass float 0-1 or other ranges. Normalization should happen before.
        # But if it's int16, we cast.
        pixel_data = pixel_data.astype(np.uint16)

    # File Meta Information
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4' # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid() # Generic implementation UID
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2' # Implicit VR Little Endian

    # Create the dataset
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Add basic standard attributes
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.Modality = modality
    ds.SeriesDescription = series_description
    
    # Dates and Times
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    dt_str = dt.strftime('%H%M%S.%f')
    ds.ContentTime = dt_str
    ds.StudyDate = dt.strftime('%Y%m%d')
    ds.SeriesDate = dt.strftime('%Y%m%d')
    ds.AcquisitionDate = dt.strftime('%Y%m%d')
    
    # UIDs
    if study_instance_uid is None:
        study_instance_uid = generate_uid()
    ds.StudyInstanceUID = study_instance_uid
    
    if series_instance_uid is None:
        series_instance_uid = generate_uid()
    ds.SeriesInstanceUID = series_instance_uid
    
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.FrameOfReferenceUID = generate_uid() # Mandatory for spatial relationship

    # Type 2 attributes (can be empty)
    ds.StudyID = "1"
    ds.SeriesNumber = 1
    ds.InstanceNumber = instance_number
    
    # Image Plane Module
    ds.PixelSpacing = [str(x) for x in pixel_spacing]
    ds.SliceThickness = str(slice_thickness)
    ds.ImagePositionPatient = [str(x) for x in image_position]
    ds.ImageOrientationPatient = [str(x) for x in image_orientation]
    
    # Pixel Data Module
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = pixel_data.shape[0]
    ds.Columns = pixel_data.shape[1]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0 # unsigned integer
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.PixelData = pixel_data.tobytes()
    
    # Transfer Syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    
    return ds

def normalize_to_uint16(data):
    """
    Normalizes a float array to 0-65535 uint16.
    """
    d_min = data.min()
    d_max = data.max()
    if d_max == d_min:
        return np.zeros_like(data, dtype=np.uint16)
    
    normalized = (data - d_min) / (d_max - d_min)
    return (normalized * 65535).astype(np.uint16)
